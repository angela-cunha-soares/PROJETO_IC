"""Etapa 4 do cronograma - Analise Descritiva e Exploratoria (EDA).

Produz a estatistica descritiva (minimo, maximo, media, mediana, quartis,
desvio-padrao, assimetria e % de faltantes) tanto da base **bruta** (23
variaveis) quanto da base **processada/imputada** (16 variaveis mantidas),
alem das figuras exploratorias (histogramas, boxplots e series temporais).

Pre-requisito: rodar ``scripts/run_preprocess.py`` antes (gera
``data/processed/dados_modelagem.csv``).

Saidas em ``reports/tables/``:
    * ``descritiva_bruta.csv``      - descritiva das 23 variaveis (dados brutos).
    * ``descritiva_processada.csv`` - descritiva das 16 variaveis mantidas (imputadas).

Saidas em ``reports/figures/``:
    * ``eda_histogramas.png``       - painel de histogramas (16 variaveis mantidas).
    * ``eda_boxplots.png``          - boxplots z-score comparando variaveis.
    * ``eda_series_temporais.png``  - series mensais de variaveis-chave.

Uso:
    python scripts/run_descritiva.py
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
from projeto_pcj.load import load_semae  # noqa: E402
from projeto_pcj.schema import FEATURES  # noqa: E402

LOG = logging.getLogger("run_descritiva")
ID_COLS = ["data", "Ano", "Mes", "mes_num", "trimestre", "estacao"]

#: Variaveis-chave para as series temporais (relevantes a irrigacao e sempre medidas).
SERIES_CHAVE = ["pH", "TURB.(FTU)", "Cond.(uS/cm)", "O.D.(ppm O2)",
                "Fe(ppm Fe)", "DUR.(ppm CaCO3)"]


def descritiva(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Tabela descritiva completa (n, faltantes, momentos e quartis) por variavel."""
    linhas = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        n = int(s.notna().sum())
        linhas.append(
            {
                "variavel": col,
                "n": n,
                "pct_faltante": round(float(s.isna().mean() * 100), 2),
                "media": round(float(s.mean()), 3) if n else np.nan,
                "desvio": round(float(s.std()), 3) if n else np.nan,
                "minimo": round(float(s.min()), 3) if n else np.nan,
                "q1": round(float(s.quantile(0.25)), 3) if n else np.nan,
                "mediana": round(float(s.median()), 3) if n else np.nan,
                "q3": round(float(s.quantile(0.75)), 3) if n else np.nan,
                "maximo": round(float(s.max()), 3) if n else np.nan,
                "assimetria": round(float(s.skew()), 3) if n > 2 else np.nan,
            }
        )
    return pd.DataFrame(linhas)


def _painel_histogramas(df: pd.DataFrame, cols: list[str], out: Path) -> None:
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    for ax, col in zip(axes.ravel(), cols):
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(serie, bins=25, color="tab:blue", alpha=0.75, edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Histogramas das variaveis mantidas (base imputada)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _painel_boxplots(df: pd.DataFrame, cols: list[str], out: Path) -> None:
    dados = df[cols].apply(lambda s: (s - s.mean()) / s.std())
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.7), 5))
    ax.boxplot([dados[c].dropna() for c in cols], labels=cols, showfliers=True)
    ax.set_title("Boxplots padronizados (z-score) - comparacao entre variaveis")
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _painel_series(df: pd.DataFrame, cols: list[str], out: Path) -> None:
    cols = [c for c in cols if c in df.columns]
    nrows = len(cols)
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 2.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(df["data"], df[col], marker=".", lw=1, color="tab:blue")
        ax.set_ylabel(col, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("data")
    fig.suptitle("Series temporais mensais - variaveis-chave (2009-2024)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config.ensure_dirs()

    LOG.info("Carregando base bruta e tabela de modelagem...")
    df = load_semae()
    modelagem = tabela_modelagem(df, estatistica="Méd.")

    desc_bruta = descritiva(modelagem, FEATURES)
    salvar_tabela(desc_bruta, "descritiva_bruta")
    LOG.info("Descritiva bruta: %d variaveis", len(desc_bruta))

    proc_path = config.PROCESSED_CSV
    if proc_path.is_file():
        proc = pd.read_csv(proc_path, encoding="utf-8")
        if "data" in proc.columns:
            proc["data"] = pd.to_datetime(proc["data"], errors="coerce")
        mantidas = [c for c in proc.columns if c not in ID_COLS]
        desc_proc = descritiva(proc, mantidas)
        salvar_tabela(desc_proc, "descritiva_processada")
        LOG.info("Descritiva processada: %d variaveis mantidas", len(mantidas))

        _painel_histogramas(proc, mantidas, config.FIGURES_DIR / "eda_histogramas.png")
        _painel_boxplots(proc, mantidas, config.FIGURES_DIR / "eda_boxplots.png")
        _painel_series(proc, SERIES_CHAVE, config.FIGURES_DIR / "eda_series_temporais.png")
        LOG.info("Figuras de EDA salvas em %s", config.FIGURES_DIR)
    else:
        LOG.warning("%s nao encontrado; rode run_preprocess.py antes.", proc_path)

    LOG.info("Analise descritiva concluida.")


if __name__ == "__main__":
    main()
