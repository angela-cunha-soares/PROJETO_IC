"""Quantifica e visualiza valores ausentes da base SEMAE consolidada.

Gera, a partir de ``data/interim/dados_organizados.csv``:

* tabela CSV com estatística de faltantes por variável e decisão de imputação
  (regras do README §4 Etapa 2);
* plots em ``reports/figures/`` para o relatório.

Plots:

1. ``missing_pct_barras.png``    — % de NaN por variável, com corte 5%.
2. ``missing_matriz.png``        — matriz observação × variável (branco = NaN).
3. ``missing_corr.png``          — correlação dos *padrões* de faltantes
   (variáveis cujos NaN aparecem juntos).
4. ``missing_temporal.png``      — % faltantes por ano para as variáveis críticas.

CSV: ``data/interim/faltantes_resumo.csv``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from projeto_pcj import load_semae  # noqa: E402
from projeto_pcj.schema import FEATURES  # noqa: E402

LOG = logging.getLogger("plot_missing")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_CSV = PROJECT_ROOT / "data" / "interim" / "faltantes_resumo.csv"

THRESH_DROP = 0.30   # > 30% NaN → excluir variável
THRESH_KNN = 0.05    # ≥ 5% NaN → KNN; < 5% → média/mediana
SKEW_SIMETRICA = 1.0  # |skew| < 1 considerada simétrica


def decidir_imputacao(pct: float, skew: float) -> str:
    if pct > THRESH_DROP:
        return "EXCLUIR"
    if pct >= THRESH_KNN:
        return "KNN"
    if abs(skew) >= SKEW_SIMETRICA:
        return "Mediana"
    return "Média"


def tabela_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói a tabela de decisão (% NaN, assimetria, regra)."""
    pct = df[FEATURES].isna().mean()
    skew = df[FEATURES].skew(numeric_only=True)
    tab = pd.DataFrame({
        "variavel": FEATURES,
        "pct_nan": (pct * 100).round(2).values,
        "skew": skew.round(2).reindex(FEATURES).values,
        "decisao": [decidir_imputacao(pct[v], skew.get(v, np.nan)) for v in FEATURES],
    }).sort_values("pct_nan", ascending=False).reset_index(drop=True)
    return tab


def plot_barras(tab: pd.DataFrame, out: Path) -> None:
    """% NaN por variável: nome ACIMA da barra e % de faltante ABAIXO da barra.

    Mostra apenas as variáveis MANTIDAS (decisão != EXCLUIR), i.e. de C.F.
    até COR. Excluir as barras vermelhas amplia a escala do eixo Y e deixa
    as diferenças entre as variáveis pequenas muito mais legíveis.
    """
    # Faixa <5% (Média e Mediana) recebe UMA cor única, pois usa-se média e
    # mediana em todas as variáveis com menos de 5% de faltantes.
    COR_EXCLUIR, COR_KNN, COR_SIMPLES = "#c0392b", "#e67e22", "#27ae60"
    cores = {"EXCLUIR": COR_EXCLUIR, "KNN": COR_KNN,
             "Mediana": COR_SIMPLES, "Média": COR_SIMPLES}
    # Remove as variáveis excluídas (>30% de faltantes) do gráfico.
    tab = tab[tab["decisao"] != "EXCLUIR"].reset_index(drop=True)
    n = len(tab)
    WIDTH = 0.62   # largura das barras (fração do passo unitário; menor = mais finas)
    MARGEM = 0.55  # folga mínima nas laterais (eixo justo, sem espaço sobrando)
    x = list(range(n))
    # Figura mais estreita => barras finas mesmo com o eixo justo (sem margens grandes).
    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.bar(x, tab["pct_nan"], color=[cores[d] for d in tab["decisao"]], width=WIDTH)
    ax.axhline(THRESH_KNN * 100, color="#e67e22", ls="--", lw=1)
    ax.set_ylabel("% de valores ausentes", fontsize=22)
    ax.set_title("Faltantes por variável mantida\n(médias mensais · 192 observações)", fontsize=22)
    ax.tick_params(axis="y", labelsize=19)
    ax.set_xticks([])
    topo = float(tab["pct_nan"].max())
    for xi, (v, p, d) in zip(x, zip(tab["variavel"], tab["pct_nan"], tab["decisao"], strict=False)):
        # nome da variável ACIMA da barra
        ax.text(xi, p + 0.4, v, ha="center", va="bottom", rotation=90, fontsize=19)
        # % de dado faltante ABAIXO da barra, na VERTICAL e alinhado com a barra
        ax.text(xi, -0.6, f"{p:.1f}%", ha="center", va="top", rotation=90, fontsize=18)
    ax.set_ylim(-6.0, topo + 14)
    ax.set_xlim(x[0] - MARGEM, x[-1] + MARGEM)
    # Legenda: apenas as regras aplicáveis às variáveis mantidas + corte 5%.
    from matplotlib.patches import Patch
    legenda = [
        Patch(facecolor=COR_KNN, label="KNN (5–30%)"),
        Patch(facecolor=COR_SIMPLES, label="Média/Mediana (<5%)"),
    ]
    legenda += [plt.Line2D([0], [0], color=COR_KNN, ls="--", label=f"≥{THRESH_KNN*100:.0f}% → KNN")]
    ax.legend(handles=legenda, loc="upper right", fontsize=17)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_matriz(df_monthly: pd.DataFrame, out: Path) -> None:
    """Matriz observação × variável (branco = NaN). Usa missingno."""
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    msno.matrix(df_monthly[FEATURES], ax=ax, sparkline=False, fontsize=9, color=(0.16, 0.5, 0.73))
    ax.set_title("Matriz de valores ausentes — preto = medido, branco = NaN")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_corr_missing(df_monthly: pd.DataFrame, out: Path) -> None:
    """Correlação entre indicadores de faltantes (padrões que aparecem juntos)."""
    mask = df_monthly[FEATURES].isna().astype(int)
    keep = mask.columns[(mask.sum() > 0) & (mask.sum() < len(mask))]  # vars com variação
    corr = mask[keep].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, cbar_kws={"label": "correlação"}, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title("Correlação entre padrões de faltantes — quais variáveis 'desaparecem juntas'")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_temporal(df_monthly: pd.DataFrame, out: Path) -> None:
    """% de meses sem medição, por ano, para variáveis críticas (5-100% NaN global)."""
    pct = df_monthly[FEATURES].isna().mean()
    vars_relevantes = [v for v in FEATURES if 0.01 < pct[v] < 1.0]
    grouped = df_monthly.groupby("Ano")[vars_relevantes].apply(lambda g: g.isna().mean() * 100)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(grouped.T, cmap="Reds", annot=True, fmt=".0f", linewidths=0.3,
                cbar_kws={"label": "% de meses sem medição"}, ax=ax)
    ax.set_title("Cobertura temporal — % de meses sem medição (por ano, por variável)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = load_semae()
    monthly = df[(df["Mes"] != "Ano") & (df["Calc"] == "Méd.")].reset_index(drop=True)
    LOG.info("Observações mensais analisadas: %d", len(monthly))

    # 1) tabela CSV
    tab = tabela_faltantes(monthly)
    tab.to_csv(OUT_CSV, index=False, encoding="utf-8")
    LOG.info("Tabela salva em %s", OUT_CSV)
    LOG.info("Decisões: %s", dict(tab["decisao"].value_counts()))

    # 2) plots
    plot_barras(tab, FIG_DIR / "missing_pct_barras.png")
    plot_matriz(monthly, FIG_DIR / "missing_matriz.png")
    plot_corr_missing(monthly, FIG_DIR / "missing_corr.png")
    plot_temporal(monthly, FIG_DIR / "missing_temporal.png")


if __name__ == "__main__":
    main()
