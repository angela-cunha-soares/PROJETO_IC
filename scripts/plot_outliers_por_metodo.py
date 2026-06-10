"""Painel de detecção de outliers POR MÉTODO, sobre dados normalizados.

Para cada método (IQR, MAD, Isolation Forest, LOF, KNN, Ensemble voto≥2/3),
mostra um scatter dos mesmos pontos projetados em 2D (PCA das variáveis
padronizadas), com os pontos marcados como outlier destacados em vermelho.

Usa a camada multivariada do projeto: variáveis com ≤ 5% de NaN, complete-case,
``StandardScaler``. IQR/MAD (univariados) são agregados por linha (união: a linha
é outlier se qualquer variável a marca), para serem comparáveis no mesmo plano.

Saída: ``reports/figures/outliers_por_metodo_normalizado.png``.

Uso:
    python scripts/plot_outliers_por_metodo.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

try:
    from pyod.models.knn import KNN as PyodKNN
except ImportError:  # pragma: no cover
    PyodKNN = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.build_features import contar_violacoes, tabela_modelagem  # noqa: E402
from projeto_pcj.load import load_semae  # noqa: E402
from projeto_pcj.schema import FEATURES  # noqa: E402

LOG = logging.getLogger("plot_outliers_por_metodo")
FIG = PROJECT_ROOT / "reports" / "figures" / "outliers_por_metodo_normalizado.png"

RNG = 42
CONTAMINATION = 0.07
MAX_MISSING_MULTI = 0.05
MAD_THRESHOLD = 3.5


def _flags_univ_uniao(M: np.ndarray, modo: str) -> np.ndarray:
    """Marca a linha se QUALQUER variável a aponta como outlier (IQR ou MAD)."""
    n, p = M.shape
    flag = np.zeros(n, dtype=bool)
    for j in range(p):
        col = M[:, j]
        if modo == "iqr":
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            flag |= (col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)
        else:  # mad
            med = np.median(col)
            mad = np.median(np.abs(col - med)) or 1e-9
            flag |= np.abs(0.6745 * (col - med) / mad) > MAD_THRESHOLD
    return flag


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    m = tabela_modelagem(load_semae())
    # Camada multivariada: variáveis com ≤ 5% de NaN, complete-case.
    feats = [c for c in FEATURES if m[c].isna().mean() <= MAX_MISSING_MULTI]
    cc = m.dropna(subset=feats).reset_index(drop=True)
    M = StandardScaler().fit_transform(cc[feats].to_numpy(float))
    LOG.info("Camada multivariada: %d obs × %d variáveis", *M.shape)

    # Flags por método
    flags: dict[str, np.ndarray] = {}
    flags["IQR (univ., união)"] = _flags_univ_uniao(M, "iqr")
    flags["MAD (univ., união)"] = _flags_univ_uniao(M, "mad")
    flags["Isolation Forest"] = (
        IsolationForest(contamination=CONTAMINATION, n_estimators=200, random_state=RNG)
        .fit_predict(M) == -1
    )
    flags["LOF"] = (
        LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION).fit_predict(M) == -1
    )
    if PyodKNN is not None:
        knn = PyodKNN(contamination=CONTAMINATION)
        knn.fit(M)
        flags["KNN (PyOD)"] = knn.labels_.astype(bool)
    else:  # pragma: no cover
        flags["KNN (PyOD)"] = np.zeros(len(M), dtype=bool)
    votos = (flags["Isolation Forest"].astype(int)
             + flags["LOF"].astype(int)
             + flags["KNN (PyOD)"].astype(int))
    flags["Ensemble (voto ≥ 2/3)"] = votos >= 2

    # Violações CONAMA (sem o Fe indicativo) — lente de domínio, não estatística.
    viol_conama = contar_violacoes(cc, excluir_indicativos=True).to_numpy()

    # Projeção 2D comum a todos os subplots (PCA dos dados padronizados)
    pca = PCA(n_components=2, random_state=RNG)
    XY = pca.fit_transform(M)
    var = pca.explained_variance_ratio_ * 100

    # 6 painéis de método (estatístico) + 1 painel CONAMA (domínio) = 7; grade 2×4.
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, (nome, mask) in zip(flat, flags.items()):
        ax.scatter(XY[~mask, 0], XY[~mask, 1], s=22, c="#2980b9",
                   alpha=0.55, label="normal")
        ax.scatter(XY[mask, 0], XY[mask, 1], s=60, c="#c0392b",
                   edgecolor="k", label="outlier")
        ax.set_title(f"{nome}  —  {int(mask.sum())} de {len(mask)}")
        ax.legend(loc="upper right", fontsize=8)

    # 7º painel: mesmos pontos coloridos pelo nº de violações CONAMA (sem Fe).
    ax_c = flat[len(flags)]
    sc = ax_c.scatter(XY[:, 0], XY[:, 1], c=viol_conama, cmap="viridis",
                      s=40, edgecolor="k", linewidth=0.3)
    ax_c.set_title(f"Violações CONAMA (sem Fe) — 0 a {int(viol_conama.max())}")
    fig.colorbar(sc, ax=ax_c, shrink=0.85, label="nº de parâmetros violados")

    # Esconde slots não usados da grade
    for ax in flat[len(flags) + 1:]:
        ax.set_visible(False)

    for ax in flat[:4]:
        ax.set_xlabel(f"PC1 ({var[0]:.0f}%)")
    for ax in (flat[0], flat[4]):
        ax.set_ylabel(f"PC2 ({var[1]:.0f}%)")

    fig.suptitle(
        "Outliers por método (estatístico) vs. violações CONAMA (domínio) — "
        "dados padronizados, projeção PCA-2D",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120, bbox_inches="tight")
    LOG.info("Figura salva em %s", FIG)


if __name__ == "__main__":
    main()
