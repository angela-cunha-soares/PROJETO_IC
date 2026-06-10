"""Calibra o parâmetro ``contamination`` (0,05–0,1) por validação cruzada (projeto §3.3).

Mede a estabilidade das flags do Isolation Forest entre folds (proxy de robustez,
Zhao et al. 2020) para cada valor da grade, na camada multivariada (10 variáveis
com ≤5% de NaN, complete-case, padronizada).

Saídas:
    * ``reports/tables/contamination_calibracao.csv``
    * ``reports/figures/contamination_calibracao.png``

Uso:
    python scripts/calibrar_contamination.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from features.build_features import tabela_modelagem  # noqa: E402
from models.reports import salvar_tabela  # noqa: E402
from preprocessing.outliers import calibrar_contamination  # noqa: E402
from projeto_pcj.load import load_semae  # noqa: E402
from projeto_pcj.schema import FEATURES  # noqa: E402

LOG = logging.getLogger("calibrar_contamination")
MAX_MISSING_MULTI = 0.05


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config.ensure_dirs()

    m = tabela_modelagem(load_semae())
    feats = [c for c in FEATURES if m[c].isna().mean() <= MAX_MISSING_MULTI]
    cc = m.dropna(subset=feats)
    X = StandardScaler().fit_transform(cc[feats].to_numpy(float))
    LOG.info("Camada multivariada: %d obs × %d variáveis", *X.shape)

    tab = calibrar_contamination(X, cv=config.RF_CV_FOLDS)
    salvar_tabela(tab.sort_values("contamination"), "contamination_calibracao")
    melhor = tab.iloc[0]
    LOG.info("Calibração (ordenada por estabilidade):\n%s", tab.to_string(index=False))
    LOG.info("Recomendado: contamination=%.2f (estabilidade Jaccard=%.3f)",
             melhor["contamination"], melhor["estabilidade_jaccard"])

    t = tab.sort_values("contamination")
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(t["contamination"], t["estabilidade_jaccard"], "o-", color="#2980b9",
             label="estabilidade (Jaccard entre folds)")
    ax1.set_xlabel("contamination")
    ax1.set_ylabel("estabilidade (Jaccard)", color="#2980b9")
    ax1.axvline(melhor["contamination"], ls="--", color="#c0392b",
                label=f"recomendado = {melhor['contamination']:.2f}")
    ax2 = ax1.twinx()
    ax2.plot(t["contamination"], t["taxa"], "s--", color="#7f8c8d", alpha=0.7,
             label="taxa observada de flags")
    ax2.set_ylabel("taxa de flags", color="#7f8c8d")
    ax1.set_title("Calibração de contamination por validação cruzada (Isolation Forest)")
    ax1.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "contamination_calibracao.png", dpi=120,
                bbox_inches="tight")
    LOG.info("Figura salva em reports/figures/contamination_calibracao.png")


if __name__ == "__main__":
    main()
