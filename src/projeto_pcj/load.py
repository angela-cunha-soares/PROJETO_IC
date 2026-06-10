"""Carregamento padronizado da base SEMAE-Piracicaba.

Função principal: :func:`load_semae`, que lê o CSV gerado por
``scripts/extrair_dados.py`` e devolve um ``DataFrame`` com tipagem e
marcadores de ausência já normalizados conforme :mod:`projeto_pcj.schema`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from projeto_pcj.schema import DTYPES, FEATURES, MISSING_MARKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "interim" / "dados_organizados.csv"

# Reconhece '30.000' como 30000 (separador de milhar) e não como 30,0 (decimal).
_THOUSAND_PATTERN = re.compile(r"^\d{1,3}(\.\d{3})+$")


def _parse_br_number(value: object) -> float:
    """Converte número em notação brasileira para ``float``, tolerando formatos mistos.

    Casos cobertos (observados na base SEMAE):

    * ``"3,23"`` → 3.23 (decimal vírgula)
    * ``"23.000"`` → 23000.0 (separador de milhar ponto)
    * ``"30.000,00"`` → 30000.0 (milhar + decimal)
    * ``"2.400.000"`` → 2400000.0
    * ``"--"``, ``"---"``, ``"-"``, ``""`` ou ``NaN`` → ``NaN``
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    s = str(value).strip()
    if s in MISSING_MARKERS:
        return float("nan")
    # Notação com vírgula: pontos são separadores de milhar
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    # Sem vírgula: pontos só são separadores de milhar se o padrão for ###.###[.###]+
    elif _THOUSAND_PATTERN.match(s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_semae(
    path: str | Path | None = None,
    *,
    dropna_all_features: bool = False,
) -> pd.DataFrame:
    """Carrega o CSV consolidado SEMAE com schema canônico aplicado.

    Trata as convenções do PDF original:

    * separador decimal vírgula (``"3,23"`` → 3.23);
    * separador de milhar ponto (``"23.000"`` → 23000.0, ``"2.400.000"`` → 2.4e6);
    * marcadores de ausência ``--``, ``---``, ``-`` mapeados para ``NaN``;
    * ``Mes`` e ``Calc`` viram ``category`` (com ordem definida em :mod:`schema`).

    Parameters
    ----------
    path
        Caminho do CSV. Se ``None``, usa ``data/interim/dados_organizados.csv``
        na raiz do projeto.
    dropna_all_features
        Se ``True``, descarta linhas em que **todas** as variáveis em
        :data:`projeto_pcj.schema.FEATURES` sejam ``NaN``.

    Returns
    -------
    pandas.DataFrame
        ``DataFrame`` com colunas ``Ano``, ``Mes``, ``Calc`` + 23 variáveis.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV não encontrado: {csv_path}. "
            "Rode antes: python scripts/extrair_dados.py"
        )

    # Lê tudo como string para depois aplicar parser brasileiro robusto.
    df = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)

    # Garantir presença de todas as colunas esperadas (preenche faltantes com NaN)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = ""

    # Reordena colunas
    df = df[["Ano", "Mes", "Calc"] + FEATURES]

    # Converte cada variável numérica via parser que tolera "30.000", "30.000,00" etc.
    for col in FEATURES:
        df[col] = df[col].map(_parse_br_number)

    # Força dtypes finais (Ano: int, Mes/Calc: category, features: float)
    df = df.astype(DTYPES)

    if dropna_all_features:
        df = df.dropna(subset=FEATURES, how="all").reset_index(drop=True)

    return df


def faltantes_por_variavel(df: pd.DataFrame) -> pd.Series:
    """Percentual de faltantes (0–1) por variável de medição, ordenado desc."""
    return df[FEATURES].isna().mean().sort_values(ascending=False)


def cobertura_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela Ano × variável com 1 = pelo menos uma medição, 0 = nenhuma.

    Útil para visualizar quando cada variável passou a ser medida (ex.: Amônia ≥ 2021).
    """
    mensais = df[df["Mes"] != "Ano"]
    return (
        mensais.groupby("Ano")[FEATURES]
        .apply(lambda g: g.notna().any().astype(int))
    )


__all__ = ["load_semae", "faltantes_por_variavel", "cobertura_temporal", "DEFAULT_CSV"]
