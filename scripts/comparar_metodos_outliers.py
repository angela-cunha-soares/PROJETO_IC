"""Compara desempenho de métodos de detecção de outliers SEM ground truth.

Como não há rótulo verdadeiro, comparamos os métodos por **critérios objetivos
internos**:

1. **Calibração** — quão perto o método chega da taxa esperada
   ``contamination = 0,07`` declarada no projeto.
2. **Concordância entre métodos** — coeficiente de Jaccard par a par e
   diagrama de Venn entre IQR, MAD e Ensemble multivariado.
3. **Distribuição temporal** — alinhamento das flags com o período da crise
   hídrica do Cantareira (2014-2016), usado como validação externa empírica.
4. **Distribuição por variável** — quais variáveis cada método "ataca" mais.

Plots gerados em ``reports/figures/``:

1. ``metodos_contagem.png``     — total de outliers por método (relativo a 192).
2. ``metodos_jaccard.png``      — matriz de Jaccard entre os métodos.
3. ``metodos_venn.png``         — Venn IQR ∪ MAD ∪ Ensemble.
4. ``metodos_temporal.png``     — % de obs marcadas por ano, por método.
5. ``metodos_por_variavel.png`` — heatmap variável × método.
6. ``metodos_resumo.png``       — tabela-resumo das métricas.

CSV: ``data/interim/comparacao_metodos.csv``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn3
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

try:
    from pyod.models.knn import KNN as PyodKNN
except ImportError:  # pragma: no cover
    PyodKNN = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from projeto_pcj import load_semae  # noqa: E402
from projeto_pcj.load import faltantes_por_variavel  # noqa: E402
from projeto_pcj.schema import FEATURES  # noqa: E402

LOG = logging.getLogger("comparar_metodos")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_CSV = PROJECT_ROOT / "data" / "interim" / "comparacao_metodos.csv"

RNG = 42
CONTAMINATION = 0.07
MAX_MISSING_MULTI = 0.05
MAD_THRESHOLD = 3.5
CRISE_INICIO, CRISE_FIM = 2014, 2016  # anos da crise Cantareira

# Paleta uniforme entre figuras
COLORS = {
    "IQR": "#3498db",
    "MAD": "#e67e22",
    "IForest": "#16a085",
    "LOF": "#8e44ad",
    "KNN": "#d35400",
    "Ensemble (voto)": "#c0392b",
}


# --- Detectores ---------------------------------------------------------------


def flag_iqr_rowwise(df: pd.DataFrame) -> np.ndarray:
    """Linha marcada se *qualquer* variável estiver fora do intervalo IQR."""
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    out = ((df.lt(q1 - 1.5 * iqr)) | (df.gt(q3 + 1.5 * iqr))).fillna(False)
    return out.any(axis=1).to_numpy()


def flag_mad_rowwise(df: pd.DataFrame, threshold: float = MAD_THRESHOLD) -> np.ndarray:
    med = df.median()
    mad = (df - med).abs().median().replace(0, np.nan)
    z = 0.6745 * (df - med) / mad
    out = (z.abs() > threshold).fillna(False)
    return out.any(axis=1).to_numpy()


def flags_multi(X: np.ndarray) -> dict[str, np.ndarray]:
    res: dict[str, np.ndarray] = {}
    res["IForest"] = IsolationForest(contamination=CONTAMINATION, n_estimators=200,
                                     random_state=RNG).fit_predict(X) == -1
    res["LOF"] = LocalOutlierFactor(contamination=CONTAMINATION).fit_predict(X) == -1
    if PyodKNN is not None:
        k = PyodKNN(contamination=CONTAMINATION); k.fit(X)
        res["KNN"] = k.labels_.astype(bool)
    return res


# --- Métricas internas --------------------------------------------------------


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


def calibracao(mask: np.ndarray, alvo: float = CONTAMINATION) -> float:
    """Distância da taxa observada à taxa-alvo (0 = perfeita)."""
    return abs(mask.mean() - alvo)


def recall_crise(mask: np.ndarray, anos: np.ndarray) -> float:
    """Fração das observações da crise (2014-2016) marcadas — proxy de recall."""
    sel = (anos >= CRISE_INICIO) & (anos <= CRISE_FIM)
    return mask[sel].sum() / sel.sum() if sel.sum() else 0.0


def precisao_relativa(mask: np.ndarray, anos: np.ndarray) -> float:
    """Fração das flags do método que caem dentro da crise — proxy de precisão."""
    sel = (anos >= CRISE_INICIO) & (anos <= CRISE_FIM)
    return (mask & sel).sum() / mask.sum() if mask.sum() else 0.0


# --- Plots --------------------------------------------------------------------


def plot_contagem(masks: dict[str, np.ndarray], n: int, out: Path) -> None:
    ser = pd.Series({k: int(v.sum()) for k, v in masks.items()})
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ser.index, ser.values, color=[COLORS.get(k, "grey") for k in ser.index])
    for b, v in zip(bars, ser.values, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v}\n({100 * v / n:.1f}%)",
                ha="center", fontsize=9)
    ax.axhline(CONTAMINATION * n, ls="--", color="grey", lw=1,
               label=f"contamination = {CONTAMINATION:.0%} ({int(CONTAMINATION * n)} obs)")
    ax.set_ylabel(f"nº de observações marcadas (de {n})")
    ax.set_title("Total de outliers detectados por método")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_jaccard(masks: dict[str, np.ndarray], out: Path) -> None:
    nomes = list(masks)
    mat = np.array([[jaccard(masks[a], masks[b]) for b in nomes] for a in nomes])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=nomes, yticklabels=nomes, cbar_kws={"label": "Jaccard"}, ax=ax)
    ax.set_title("Concordância entre métodos (1 = idênticos, 0 = disjuntos)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_venn(iqr: np.ndarray, mad: np.ndarray, ensemble: np.ndarray, out: Path) -> None:
    sets = {
        "IQR": set(np.where(iqr)[0]),
        "MAD": set(np.where(mad)[0]),
        "Ensemble": set(np.where(ensemble)[0]),
    }
    fig, ax = plt.subplots(figsize=(8, 7))
    venn3(subsets=[sets["IQR"], sets["MAD"], sets["Ensemble"]],
          set_labels=("IQR (univariado)", "MAD (univariado)", "Ensemble (multivariado)"),
          set_colors=(COLORS["IQR"], COLORS["MAD"], COLORS["Ensemble (voto)"]),
          ax=ax)
    ax.set_title("Sobreposição entre os três métodos principais")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_temporal(masks: dict[str, np.ndarray], anos: np.ndarray, out: Path) -> None:
    df = pd.DataFrame(masks, dtype=int)
    df["Ano"] = anos
    rate = df.groupby("Ano").mean() * 100
    fig, ax = plt.subplots(figsize=(13, 6))
    for col in rate.columns:
        ax.plot(rate.index, rate[col], marker="o", lw=2, label=col, color=COLORS.get(col, "grey"))
    ax.axvspan(CRISE_INICIO - 0.5, CRISE_FIM + 0.5, color="red", alpha=0.1,
               label=f"Crise Cantareira ({CRISE_INICIO}-{CRISE_FIM})")
    ax.set_ylabel("% de meses marcados como outlier")
    ax.set_xlabel("Ano")
    ax.set_title("Distribuição temporal das flags por método")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_por_variavel(df_uni: pd.DataFrame, masks_uni: dict[str, np.ndarray],
                      out: Path) -> None:
    """Quantas vezes cada variável "contribuiu" para a flag de cada método univariado."""
    q1, q3 = df_uni.quantile(0.25), df_uni.quantile(0.75)
    iqr = q3 - q1
    iqr_per_var = ((df_uni.lt(q1 - 1.5 * iqr)) | (df_uni.gt(q3 + 1.5 * iqr))).fillna(False).sum()

    med = df_uni.median()
    mad_v = (df_uni - med).abs().median().replace(0, np.nan)
    z = 0.6745 * (df_uni - med) / mad_v
    mad_per_var = (z.abs() > MAD_THRESHOLD).fillna(False).sum()

    tab = pd.DataFrame({"IQR": iqr_per_var, "MAD": mad_per_var})
    tab = tab.sort_values("IQR", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    tab.plot.bar(ax=ax, width=0.8, color=[COLORS["IQR"], COLORS["MAD"]])
    ax.set_ylabel("nº de outliers detectados")
    ax.set_title("Quais variáveis cada método sinaliza mais")
    ax.tick_params(axis="x", rotation=80)
    ax.legend(title="Método")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_resumo(tab: pd.DataFrame, out: Path) -> None:
    """Tabela visual das métricas internas — pronta para colar no relatório."""
    fig, ax = plt.subplots(figsize=(11, max(2, 0.45 * len(tab) + 1.5)))
    ax.axis("off")
    cell_text = tab.round(3).astype(str).values
    table = ax.table(cellText=cell_text, colLabels=tab.columns, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    # Destaca a linha-vencedora (menor distância à calibração)
    melhor = tab["dist_calibração"].idxmin()
    for col in range(len(tab.columns)):
        table[(melhor + 1, col)].set_facecolor("#fcefb4")
    ax.set_title("Comparação interna de métodos — destaque = melhor calibração", pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


# --- Pipeline -----------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = load_semae()
    monthly = df[(df["Mes"] != "Ano") & (df["Calc"] == "Méd.")].reset_index(drop=True)
    anos = monthly["Ano"].to_numpy()
    n = len(monthly)
    LOG.info("Observações: %d", n)

    X_uni = monthly[FEATURES]

    # ---- Univariados (operam direto, NaN ignorado) ----
    iqr_flag = flag_iqr_rowwise(X_uni)
    mad_flag = flag_mad_rowwise(X_uni)

    # ---- Multivariados (complete-cases das vars ≤ 5% NaN) ----
    falt = faltantes_por_variavel(monthly)
    feats_multi = [c for c in FEATURES if falt[c] <= MAX_MISSING_MULTI]
    X_multi = monthly[feats_multi]
    complete = X_multi.notna().all(axis=1).to_numpy()
    X_std = StandardScaler().fit_transform(X_multi.loc[complete].to_numpy())
    multi_subset = flags_multi(X_std)

    # Devolve cada flag ao espaço total
    flags_full: dict[str, np.ndarray] = {"IQR": iqr_flag, "MAD": mad_flag}
    for k, m_sub in multi_subset.items():
        v = np.zeros(n, dtype=bool); v[complete] = m_sub; flags_full[k] = v
    # Voto majoritário do ensemble (no subset)
    voto_sub = np.vstack(list(multi_subset.values())).sum(axis=0) >= 2
    voto = np.zeros(n, dtype=bool); voto[complete] = voto_sub
    flags_full["Ensemble (voto)"] = voto
    LOG.info("Totais por método: %s", {k: int(v.sum()) for k, v in flags_full.items()})

    # ---- Métricas internas ----
    linhas = []
    for nome, mask in flags_full.items():
        linhas.append({
            "método": nome,
            "n_flags": int(mask.sum()),
            "taxa_observada": round(mask.mean(), 4),
            "dist_calibração": round(calibracao(mask), 4),
            "recall_crise_2014_2016": round(recall_crise(mask, anos), 3),
            "precisão_crise": round(precisao_relativa(mask, anos), 3),
        })
    tab = pd.DataFrame(linhas)

    # Concordância média de cada método com os outros (Jaccard)
    nomes = list(flags_full)
    jac = pd.DataFrame(
        [[jaccard(flags_full[a], flags_full[b]) for b in nomes] for a in nomes],
        index=nomes, columns=nomes,
    )
    tab["concordância_média"] = tab["método"].map(
        lambda m: round(jac.loc[m].drop(m).mean(), 3)
    )
    tab = tab.set_index("método").reset_index()
    tab.to_csv(OUT_CSV, index=False, encoding="utf-8")
    LOG.info("Tabela salva em %s", OUT_CSV)
    LOG.info("\n%s", tab.to_string(index=False))

    # ---- Plots ----
    plot_contagem(flags_full, n, FIG_DIR / "metodos_contagem.png")
    plot_jaccard(flags_full, FIG_DIR / "metodos_jaccard.png")
    plot_venn(iqr_flag, mad_flag, voto, FIG_DIR / "metodos_venn.png")
    plot_temporal(flags_full, anos, FIG_DIR / "metodos_temporal.png")
    plot_por_variavel(X_uni, {"IQR": iqr_flag, "MAD": mad_flag}, FIG_DIR / "metodos_por_variavel.png")
    plot_resumo(tab, FIG_DIR / "metodos_resumo.png")


if __name__ == "__main__":
    main()
