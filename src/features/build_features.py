"""Engenharia de variáveis e rotulagem regulatória.

Duas responsabilidades:

1. :func:`tabela_modelagem` — converte o CSV "longo" (Mín./Méd./Máx. por mês)
   na tabela de modelagem: uma linha por mês usando a estatística ``Méd.``,
   com coluna de data e índices temporais (mês, trimestre, estação seca/úmida).
2. :func:`rotular_conama` — deriva o rótulo ``adequada/inadequada`` para o
   Random Forest a partir dos limites da Resolução CONAMA 357/2005, Classe 2
   (ADR-005), com salvaguarda contra vazamento de dados (*data leakage*).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from projeto_pcj.schema import CONAMA_357_CLASSE2, FEATURES

#: Ordem dos meses → número, para construir datas e índices sazonais.
_MES_NUM = {
    "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12,
}
#: Estação úmida (chuvas) no clima subtropical do PCJ: out–mar.
_MESES_UMIDOS = {10, 11, 12, 1, 2, 3}

#: Parâmetros cujo limite CONAMA é por fração diferente da medida pelo SEMAE,
#: tornando a violação apenas *indicativa* (não conclusiva). O Ferro é o caso:
#: a norma limita Fe DISSOLVIDO (0,3 mg/L), mas o SEMAE mede Fe TOTAL → violação
#: superestimada. Ver docs/dicionario_de_dados.md e schema.CONAMA_357_CLASSE2.
PARAMS_INDICATIVOS: list[str] = ["Fe(ppm Fe)"]


def tabela_modelagem(df: pd.DataFrame, estatistica: str = "Méd.") -> pd.DataFrame:
    """Filtra a estatística mensal escolhida e adiciona índices temporais.

    Parameters
    ----------
    df
        Saída de :func:`projeto_pcj.load.load_semae`.
    estatistica
        ``"Méd."`` (padrão), ``"Mín."`` ou ``"Máx."``.

    Returns
    -------
    DataFrame
        Uma linha por mês×ano (exclui as linhas de resumo ``Ano``), com colunas
        ``data``, ``mes_num``, ``trimestre``, ``estacao`` e as variáveis físico-químicas.
    """
    sub = df[(df["Calc"] == estatistica) & (df["Mes"] != "Ano")].copy()
    sub["mes_num"] = sub["Mes"].map(_MES_NUM).astype(int)
    sub["data"] = pd.to_datetime(
        dict(year=sub["Ano"], month=sub["mes_num"], day=1)
    )
    sub["trimestre"] = sub["data"].dt.quarter
    sub["estacao"] = np.where(sub["mes_num"].isin(_MESES_UMIDOS), "úmida", "seca")
    cols = ["data", "Ano", "Mes", "mes_num", "trimestre", "estacao"] + FEATURES
    return sub[cols].sort_values("data").reset_index(drop=True)


def violacoes_conama(
    df: pd.DataFrame,
    features: list[str] | None = None,
    *,
    excluir_indicativos: bool = False,
) -> pd.DataFrame:
    """Matriz booleana de violação por linha × variável (CONAMA 357/2005 Classe 2).

    ``True`` indica que o valor extrapola o limite legal da variável.
    Variáveis sem limite definido no schema são ignoradas.

    Parameters
    ----------
    excluir_indicativos
        Se ``True``, ignora os parâmetros de :data:`PARAMS_INDICATIVOS` (ex.: Fe,
        cujo limite é por fração diferente da medida) — evita superestimar as
        violações. Padrão ``False`` (mantém o comportamento conservador da norma).
    """
    features = features or [c for c in FEATURES if c in CONAMA_357_CLASSE2]
    if excluir_indicativos:
        features = [c for c in features if c not in PARAMS_INDICATIVOS]
    viol = pd.DataFrame(index=df.index)
    for col in features:
        if col not in CONAMA_357_CLASSE2 or col not in df.columns:
            continue
        low, high = CONAMA_357_CLASSE2[col]
        flag = pd.Series(False, index=df.index)
        if low is not None:
            flag |= df[col] < low
        if high is not None:
            flag |= df[col] > high
        viol[col] = flag.fillna(False)
    return viol


def contar_violacoes(
    df: pd.DataFrame,
    features: list[str] | None = None,
    *,
    excluir_indicativos: bool = False,
) -> pd.Series:
    """Número de parâmetros CONAMA violados por linha (carga de não conformidade).

    ``excluir_indicativos=True`` desconta os parâmetros de :data:`PARAMS_INDICATIVOS`
    (ex.: Fe total vs. dissolvido).
    """
    return (
        violacoes_conama(df, features, excluir_indicativos=excluir_indicativos)
        .sum(axis=1)
        .rename("n_violacoes")
    )


def rotular_conama(
    df: pd.DataFrame,
    *,
    features_rotulo: list[str] | None = None,
) -> tuple[pd.Series, list[str]]:
    """Rótulo binário ``inadequada`` (1) / ``adequada`` (0) — união das violações.

    Uma amostra é **inadequada** se violar o limite de **qualquer** variável
    normatizada (critério sensível, ADR-005).

    Atenção: em água **bruta** de rio, este rótulo costuma ser degenerado
    (quase tudo "inadequado"), pois a CONAMA 357/2005 Classe 2 normatiza o
    corpo d'água, não a água já tratada. Para uma tarefa supervisionada
    balanceada use :func:`rotular_severidade`.

    Returns
    -------
    (rotulo, variaveis_usadas)
        ``rotulo`` é uma ``Series`` 0/1; ``variaveis_usadas`` lista as colunas
        que entraram na regra — elas **não devem** ser usadas como features do
        classificador (evita *data leakage*, ADR-005).
    """
    viol = violacoes_conama(df, features_rotulo)
    rotulo = viol.any(axis=1).astype(int)
    rotulo.name = "inadequada"
    return rotulo, list(viol.columns)


def rotular_severidade(
    df: pd.DataFrame,
    *,
    features_rotulo: list[str] | None = None,
    limiar: float | None = None,
) -> tuple[pd.Series, list[str]]:
    """Rótulo binário de severidade: ``alta`` (1) / ``baixa`` (0) carga de violações.

    Conta quantos parâmetros CONAMA cada amostra viola e binariza pela mediana
    (ou por ``limiar`` explícito). Em água bruta — onde a união das violações é
    degenerada — este alvo é balanceado e ainda ancorado na norma legal.

    Returns
    -------
    (rotulo, variaveis_usadas)
        Mesmos cuidados anti-leakage de :func:`rotular_conama`.
    """
    viol = violacoes_conama(df, features_rotulo)
    n = viol.sum(axis=1)
    corte = limiar if limiar is not None else float(n.median())
    rotulo = (n > corte).astype(int)
    rotulo.name = "severidade_alta"
    return rotulo, list(viol.columns)


def features_sem_vazamento(
    features_modelo: list[str], variaveis_rotulo: list[str]
) -> list[str]:
    """Remove de ``features_modelo`` as variáveis usadas na rotulagem (anti-leakage)."""
    proibidas = set(variaveis_rotulo)
    return [c for c in features_modelo if c not in proibidas]


__all__ = [
    "tabela_modelagem", "violacoes_conama", "contar_violacoes",
    "rotular_conama", "rotular_severidade", "features_sem_vazamento",
    "PARAMS_INDICATIVOS",
]
