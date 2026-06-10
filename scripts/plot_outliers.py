"""Identifica outliers SEM imputação, em duas camadas, e gera as figuras da Etapa 4.

Camada UNIVARIADA (todas as 23 variáveis em todas as 192 linhas mensais ``Méd.``,
``NaN`` ignorado):

* IQR clássico — ``x < Q1 - 1.5·IQR`` ou ``x > Q3 + 1.5·IQR``.
* Z-score robusto via MAD (Iglewicz & Hoaglin, 1993) —
  ``|0.6745 · (x - mediana) / MAD| > 3.5``.

Camada MULTIVARIADA (apenas variáveis com ≤ 5% de ``NaN``, apenas observações
complete-case, padronizadas com ``StandardScaler``):

* Isolation Forest, Local Outlier Factor, PyOD-KNN — todos com
  ``contamination = 0,07``.
* Voto majoritário (≥ 2 dos 3 métodos marcam).

Cada observação é então classificada como:

============= =========================================================
Categoria     Definição
============= =========================================================
**Forte**     Marcada em ambas as camadas (univariada + multivariada).
**Só uni**    Marcada só na univariada.
**Só multi**  Marcada só na multivariada.
**Incompleta** Tem ``NaN`` nas variáveis multivariadas — sem veredito multi.
**Limpa**     Não marcada em nenhuma camada.
============= =========================================================

Plots gerados em ``reports/figures/``:

1. ``outliers_uni_boxplots.png``    — boxplots padronizados (todas 23 vars).
2. ``outliers_uni_contagem.png``    — flags por variável (IQR vs MAD).
3. ``outliers_multi_contagem.png``  — flags por método multivariado.
4. ``outliers_multi_heatmap.png``   — obs × método (complete cases).
5. ``outliers_classificacao.png``   — nº de observações por categoria.
6. ``outliers_serie_temporal.png``  — séries com pontos coloridos por categoria.
7. ``outliers_scatter_pH_TURB.png`` — pH × Turbidez por categoria.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

LOG = logging.getLogger("plot_outliers")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

RNG = 42
CONTAMINATION = 0.07
MAX_MISSING_MULTI = 0.05  # ≤ 5% NaN para entrar na camada multivariada
MAD_THRESHOLD = 3.5       # Iglewicz & Hoaglin (1993)

# Paleta consistente entre figuras
COLOR = {
    "Forte": "#c0392b",       # vermelho escuro
    "Só uni": "#e67e22",      # laranja
    "Só multi": "#8e44ad",    # roxo
    "Incompleta": "#7f8c8d",  # cinza
    "Limpa": "#2980b9",       # azul
}

# --- Detectores ---------------------------------------------------------------


def flags_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """``True`` onde o valor cai fora de ``[Q1 - 1.5·IQR, Q3 + 1.5·IQR]``. NaN → False."""
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    return ((df.lt(q1 - 1.5 * iqr)) | (df.gt(q3 + 1.5 * iqr))).fillna(False)


def flags_mad(df: pd.DataFrame, threshold: float = MAD_THRESHOLD) -> pd.DataFrame:
    """Z-score robusto via MAD; ``True`` onde |z| > threshold. NaN → False."""
    med = df.median()
    mad = (df - med).abs().median()
    # Evita divisão por zero em variáveis quase constantes.
    mad = mad.replace(0, np.nan)
    z = 0.6745 * (df - med) / mad
    return (z.abs() > threshold).fillna(False)


def flags_ensemble_multi(X: np.ndarray) -> dict[str, np.ndarray]:
    """Roda IForest, LOF e PyOD-KNN; cada um devolve máscara booleana ``True`` = outlier."""
    out: dict[str, np.ndarray] = {}
    iforest = IsolationForest(contamination=CONTAMINATION, n_estimators=200, random_state=RNG)
    out["IForest"] = iforest.fit_predict(X) == -1
    lof = LocalOutlierFactor(contamination=CONTAMINATION)
    out["LOF"] = lof.fit_predict(X) == -1
    if PyodKNN is not None:
        knn = PyodKNN(contamination=CONTAMINATION)
        knn.fit(X)
        out["KNN"] = knn.labels_.astype(bool)
    else:
        LOG.warning("pyod ausente — KNN não rodou")
    return out


# --- Plots --------------------------------------------------------------------


def plot_uni_boxplots(df: pd.DataFrame, out: Path) -> None:
    """Boxplot padronizado (z) de todas as variáveis; pontos vermelhos = IQR outliers."""
    z = (df - df.mean()) / df.std()
    fig, ax = plt.subplots(figsize=(13, 6))
    z.boxplot(ax=ax, rot=80, grid=False, sym="r+", widths=0.6)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Camada univariada — boxplots padronizados (z-score) com outliers IQR")
    ax.set_ylabel("z-score")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_uni_contagem(f_iqr: pd.DataFrame, f_mad: pd.DataFrame, out: Path) -> None:
    """Barras: nº de flags por variável (IQR vs MAD)."""
    cont = pd.DataFrame({"IQR (1.5·IQR)": f_iqr.sum(), "MAD (|z|>3.5)": f_mad.sum()})
    cont = cont.sort_values("IQR (1.5·IQR)", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    cont.plot.bar(ax=ax, width=0.8, color=["#3498db", "#e67e22"])
    ax.set_title("Camada univariada — outliers por variável (NaN não conta)")
    ax.set_ylabel("nº de observações marcadas")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=80)
    ax.legend(title="Método")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_multi_contagem(flags_multi: dict[str, np.ndarray], voto: np.ndarray, n_total: int, out: Path) -> None:
    """Barras: total de outliers por método multivariado + voto majoritário."""
    dados = {nome: int(m.sum()) for nome, m in flags_multi.items()}
    dados["Voto ≥ 2/3"] = int(voto.sum())
    s = pd.Series(dados)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = s.plot.bar(ax=ax, color=["#3498db", "#3498db", "#3498db", "#c0392b"], width=0.65)
    for i, v in enumerate(s):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=10)
    ax.set_title(f"Camada multivariada — outliers ({n_total} complete cases, contamination = 0,07)")
    ax.set_ylabel("nº de observações marcadas")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_multi_heatmap(flags_multi: dict[str, np.ndarray], voto: np.ndarray, out: Path) -> None:
    """Heatmap observação × método (apenas observações complete-case)."""
    M = pd.DataFrame(flags_multi)
    M["Voto ≥ 2/3"] = voto
    fig, ax = plt.subplots(figsize=(6, max(5, len(M) * 0.08)))
    sns.heatmap(M.astype(int), cmap="Reds", cbar=False, linewidths=0.1, yticklabels=False, ax=ax)
    ax.set_title("Camada multivariada — obs × método (1 = flag)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_classificacao(categorias: pd.Series, out: Path) -> None:
    """Barras horizontais: nº de obs por categoria, com a paleta global."""
    counts = categorias.value_counts().reindex(list(COLOR.keys())).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.barh(ax=ax, color=[COLOR[c] for c in counts.index])
    for i, v in enumerate(counts):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=10)
    ax.set_title("Classificação final por observação (sem imputação)")
    ax.set_xlabel("nº de observações")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_serie_temporal(df: pd.DataFrame, datas: pd.Series, categorias: pd.Series, vars_destaque: list[str], out: Path) -> None:
    """Séries temporais com pontos coloridos por categoria."""
    n = len(vars_destaque)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, var in zip(axes, vars_destaque, strict=False):
        ax.plot(datas, df[var], color="lightgrey", lw=1, zorder=1)
        for cat, cor in COLOR.items():
            mask = (categorias == cat) & df[var].notna().to_numpy()
            if mask.any():
                # Pontos limpos ficam menores e atrás
                size = 18 if cat == "Limpa" else 50
                z = 2 if cat == "Limpa" else 4
                ax.scatter(datas[mask], df.loc[mask, var], c=cor, s=size, label=cat,
                           zorder=z, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(var.split("(")[0].strip())
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=5)
    axes[-1].set_xlabel("Data (mensal, média)")
    fig.suptitle("Séries temporais — classificação por categoria", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_scatter(df: pd.DataFrame, categorias: pd.Series, x: str, y: str, out: Path) -> None:
    """Dispersão x × y colorida por categoria."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for cat, cor in COLOR.items():
        mask = (categorias == cat) & df[x].notna() & df[y].notna()
        if mask.any():
            size = 30 if cat == "Limpa" else 70
            ax.scatter(df.loc[mask, x], df.loc[mask, y], c=cor, s=size, label=cat,
                       alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} × {y} — classificação por observação")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


# --- Pipeline -----------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info("Carregando dados…")
    df = load_semae()

    # 1) Subset: mensais (descarta resumo Ano), estatística Méd.
    monthly = df[(df["Mes"] != "Ano") & (df["Calc"] == "Méd.")].reset_index(drop=True)
    LOG.info("Observações mensais (Méd.): %d", len(monthly))
    X_uni = monthly[FEATURES]

    # 2) Camada univariada (todas as 23 variáveis, NaN ignorado)
    f_iqr = flags_iqr(X_uni)
    f_mad = flags_mad(X_uni)
    flagged_uni = (f_iqr | f_mad).any(axis=1)
    LOG.info("Univariada: %d/%d linhas marcadas", flagged_uni.sum(), len(flagged_uni))

    # 3) Camada multivariada (≤ 5% NaN; complete cases; padronizado)
    falt = faltantes_por_variavel(monthly)
    feats_multi = [c for c in FEATURES if falt[c] <= MAX_MISSING_MULTI]
    X_multi = monthly[feats_multi]
    complete_mask = X_multi.notna().all(axis=1)
    X_complete = X_multi.loc[complete_mask].to_numpy()
    X_std = StandardScaler().fit_transform(X_complete)
    LOG.info(
        "Multivariada: %d variáveis com ≤ %.0f%% NaN; %d/%d linhas complete-case",
        len(feats_multi), 100 * MAX_MISSING_MULTI, complete_mask.sum(), len(complete_mask),
    )

    flags_multi = flags_ensemble_multi(X_std)
    voto_multi = np.vstack(list(flags_multi.values())).sum(axis=0) >= 2
    LOG.info("Multivariada: %d outliers pelo voto ≥ 2/3", int(voto_multi.sum()))

    # Devolve o flag multivariado ao espaço de todas as observações
    flagged_multi = pd.Series(False, index=monthly.index)
    flagged_multi.loc[complete_mask] = voto_multi

    # 4) Classificação final
    categorias = pd.Series("Limpa", index=monthly.index, dtype="object")
    categorias[flagged_uni & flagged_multi] = "Forte"
    categorias[flagged_uni & ~flagged_multi & complete_mask] = "Só uni"
    categorias[~flagged_uni & flagged_multi] = "Só multi"
    categorias[~complete_mask & ~flagged_uni] = "Incompleta"
    categorias[~complete_mask & flagged_uni] = "Só uni"  # univ ainda vale sem complete-case
    LOG.info("Classificação: %s", dict(categorias.value_counts()))

    # 5) Plots
    plot_uni_boxplots(X_uni, FIG_DIR / "outliers_uni_boxplots.png")
    plot_uni_contagem(f_iqr, f_mad, FIG_DIR / "outliers_uni_contagem.png")
    plot_multi_contagem(flags_multi, voto_multi, n_total=int(complete_mask.sum()),
                        out=FIG_DIR / "outliers_multi_contagem.png")
    plot_multi_heatmap(flags_multi, voto_multi, FIG_DIR / "outliers_multi_heatmap.png")
    plot_classificacao(categorias, FIG_DIR / "outliers_classificacao.png")

    meses_map = {m: i + 1 for i, m in enumerate(
        ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    )}
    datas = pd.to_datetime(
        {"year": monthly["Ano"], "month": monthly["Mes"].map(meses_map), "day": 15}
    )
    vars_destaque = [v for v in ["pH", "TURB.(FTU)", "Cond.(uS/cm)", "Fe(ppm Fe)", "CLOROFILA(ug/l)"]
                     if v in FEATURES]
    plot_serie_temporal(X_uni, datas, categorias, vars_destaque,
                        FIG_DIR / "outliers_serie_temporal.png")
    plot_scatter(X_uni, categorias, "pH", "TURB.(FTU)", FIG_DIR / "outliers_scatter_pH_TURB.png")

    # 6) Anexa as flags ao CSV interim para uso nas etapas seguintes
    out_csv = PROJECT_ROOT / "data" / "interim" / "outliers_classificacao.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    saida = monthly[["Ano", "Mes"]].copy()
    saida["categoria"] = categorias
    saida["uni_iqr"] = f_iqr.any(axis=1).to_numpy()
    saida["uni_mad"] = f_mad.any(axis=1).to_numpy()
    saida["multi_voto"] = flagged_multi.to_numpy()
    saida.to_csv(out_csv, index=False, encoding="utf-8")
    LOG.info("Tabela de classificação salva em %s", out_csv)


if __name__ == "__main__":
    main()
