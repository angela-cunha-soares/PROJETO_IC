# -*- coding: utf-8 -*-
"""Validação da imputação por mascaramento (hold-out) — comparação de métodos.

Segue exatamente a ordem metodológica declarada no projeto:

    1. Identifica o percentual de dados faltantes por variável.
    2. Classifica cada variável pela regra:
         > 30%      -> EXCLUIR
         < 5%       -> Média (simétrica) ou Mediana (assimétrica)
         5% – 30%   -> KNN
    3. Valida a imputação escondendo (MCAR) parte das células que de fato
       foram medidas, imputando por Média, Mediana e KNN, e medindo o erro
       (RMSE/MAE/nRMSE) apenas nas células mascaradas — cujo valor verdadeiro
       conhecemos. Assim descobre-se, empiricamente, qual método "se sai
       melhor" em cada variável, comparando-o com o método aplicado pela regra.

Pré-requisito: apenas a base organizada
``data/interim/dados_organizados.csv`` (nada mais).

Saídas em ``reports/tables/``:
    * ``imputacao_percentual_faltantes.csv`` — % de faltantes + faixa + método da regra.
    * ``imputacao_benchmark.csv``            — RMSE/MAE/nRMSE por variável × método.
    * ``imputacao_melhor_metodo.csv``        — melhor método (menor nRMSE) por variável.

Saída em ``reports/figures/``:
    * ``imputacao_benchmark_nrmse.png``      — nRMSE por variável e método.

Uso:
    python scripts/validar_imputacao.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from features.build_features import tabela_modelagem  # noqa: E402
from models.reports import salvar_tabela  # noqa: E402
from preprocessing.missing import (  # noqa: E402
    benchmark_imputacao, imputar, planejar_imputacao, resumo_faltantes,
    validar_imputacao as validar_distribuicao,
)
from projeto_pcj.load import load_semae  # noqa: E402
from projeto_pcj.schema import FEATURES  # noqa: E402

LOG = logging.getLogger("validar_imputacao")

FAIXA = {
    "EXCLUIR": "> 30% (excluída)",
    "Média": "< 5%",
    "Mediana": "< 5%",
    "KNN": "5% – 30%",
}


def _figura_nrmse(bench: pd.DataFrame, ordem: list[str], out: Path) -> None:
    metodos = ["media", "mediana", "knn"]
    cores = {"media": "tab:blue", "mediana": "tab:orange", "knn": "tab:green"}
    piv = bench.pivot(index="variavel", columns="metodo", values="nrmse")
    piv = piv.reindex(ordem)
    x = np.arange(len(piv))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(9, len(piv) * 0.7), 5))
    for i, m in enumerate(metodos):
        if m in piv.columns:
            ax.bar(x + (i - 1) * w, piv[m].to_numpy(), width=w,
                   label=m, color=cores[m])
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=90, fontsize=8)
    ax.set_ylabel("nRMSE (RMSE ÷ desvio) — menor é melhor")
    ax.set_title("Validação da imputação por mascaramento: erro por variável e método")
    ax.legend(title="método")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config.ensure_dirs()

    LOG.info("1) Carregando base e montando tabela de modelagem…")
    df = load_semae()
    modelagem = tabela_modelagem(df, estatistica="Méd.")

    # 1) Percentual de faltantes por variável + decisão da regra
    resumo = resumo_faltantes(modelagem, FEATURES)
    resumo["faixa"] = resumo["decisao"].map(FAIXA)
    salvar_tabela(
        resumo.rename(columns={"pct_nan": "pct_faltante", "decisao": "metodo_regra"}),
        "imputacao_percentual_faltantes",
    )
    plano = planejar_imputacao(modelagem, FEATURES)
    LOG.info("Excluídas (>30%%): %s", plano.excluir)
    LOG.info("Mantidas p/ validação (%d): %s", len(plano.manter), plano.manter)

    # Descarta meses sem NENHUMA medição (ex.: Dez/2024) apenas para os cálculos
    # numéricos — as decisões acima já foram tomadas sobre a grade completa.
    medido = modelagem.dropna(subset=FEATURES, how="all").reset_index(drop=True)
    n_vazios = len(modelagem) - len(medido)
    if n_vazios:
        LOG.info("Meses sem nenhuma medição descartados na validação: %d", n_vazios)

    # Teste 1) Mascaramento (hold-out) comparando média, mediana e KNN
    LOG.info("Teste 1) Mascaramento (30 repetições, 15%% de máscara)…")
    bench = benchmark_imputacao(medido, plano.manter,
                                frac_mascara=0.15, n_repeticoes=30)
    salvar_tabela(bench, "imputacao_benchmark")

    # Melhor método (menor nRMSE) por variável + método aplicado pela regra
    metodo_regra = {}
    for _, r in resumo.iterrows():
        metodo_regra[r["variavel"]] = r["decisao"]
    def _vencedor(metrica: str) -> pd.DataFrame:
        return (
            bench.sort_values(metrica)
            .groupby("variavel", as_index=False)
            .first()[["variavel", "metodo"]]
            .rename(columns={"metodo": f"melhor_{metrica}"})
        )

    melhor = _vencedor("nrmse").merge(_vencedor("mae"), on="variavel")
    melhor["metodo_regra"] = melhor["variavel"].map(
        lambda v: {"Média": "media", "Mediana": "mediana", "KNN": "knn"}.get(
            metodo_regra.get(v, ""), metodo_regra.get(v, ""))
    )
    # Erro do método efetivamente APLICADO pela regra em cada variável — valida
    # o método escolhido (não só a comparação entre métodos).
    idx = bench.set_index(["variavel", "metodo"])
    def _erro(v, m, col):
        try:
            return float(idx.loc[(v, m), col])
        except KeyError:
            return float("nan")
    melhor["nrmse_aplicado"] = [_erro(v, m, "nrmse")
                                for v, m in zip(melhor["variavel"], melhor["metodo_regra"])]
    melhor["mae_aplicado"] = [_erro(v, m, "mae")
                              for v, m in zip(melhor["variavel"], melhor["metodo_regra"])]
    # nRMSE favorece a média (minimiza erro quadrático); MAE favorece a mediana
    # (robusta a assimetria). Comparamos ambos com o método aplicado pela regra.
    melhor["regra_confirma_mae"] = melhor["melhor_mae"] == melhor["metodo_regra"]
    salvar_tabela(melhor, "imputacao_melhor_metodo")

    # Figura
    ordem = [c for c in plano.manter if c in set(bench["variavel"])]
    _figura_nrmse(bench, ordem, config.FIGURES_DIR / "imputacao_benchmark_nrmse.png")

    LOG.info("Vencedor por variável (nRMSE favorece média; MAE favorece mediana):"
             "\n%s", melhor.to_string(index=False))
    n_ok = int(melhor["regra_confirma_mae"].sum())
    LOG.info("Pela métrica robusta (MAE), o método da regra coincide com o "
             "vencedor empírico em %d de %d variáveis mantidas.", n_ok, len(melhor))

    # Teste 2) Distribuição (média/desvio) antes vs. depois da imputação real
    LOG.info("Teste 2) Comparando média/desvio antes e depois da imputação…")
    imputado, _ = imputar(medido, FEATURES, plano=plano)
    dist = validar_distribuicao(medido, imputado, plano.manter)
    salvar_tabela(dist, "validacao_imputacao")
    LOG.info("Maior |Δ desvio| entre as mantidas: %.2f%%",
             float(dist["delta_std_%"].abs().max()))
    LOG.info("Validação concluída (2 testes: mascaramento + antes/depois).")


if __name__ == "__main__":
    main()
