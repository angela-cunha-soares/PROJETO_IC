"""Avalia 6 métodos univariados de detecção de outliers POR VARIÁVEL.

Compara, para cada uma das 23 variáveis, os métodos:

* ``IQR_1.5`` — regra clássica de Tukey (1,5·IQR).
* ``IQR_3.0`` — extremos só (3·IQR).
* ``Z-score`` — paramétrico, assume normalidade.
* ``MAD`` — z-score robusto (Iglewicz & Hoaglin, 1993).
* ``Percentil 1-99`` — fora do intervalo ``[P₁, P₉₉]``.
* ``Tukey adjusted`` — boxplot ajustado por assimetria via *medcouple*
  (Hubert & Vandervieren, 2008).

Critério de "melhor" por variável:

* **Com limite CONAMA 357/2005 Classe 2** (15 variáveis em :mod:`schema`):
  precisão, recall e F1 contra a "violação CONAMA" como ground truth.
  Vence o método com maior F1.
* **Sem limite CONAMA** (8 variáveis): estabilidade via concordância média
  (Jaccard) com os outros métodos univariados — proxy de consenso.
  Vence o método com maior concordância média.

⚠️ Violação CONAMA não é idêntica a outlier estatístico — é apenas um proxy
*de poluição regulatoriamente relevante*. Documente esse distinção no relatório.

Saídas em ``reports/figures/``:

1. ``melhor_metodo_heatmap.png``       — variável × método, cor = F1 ou Jaccard.
2. ``melhor_metodo_ranking.png``       — método-vencedor por variável.
3. ``melhor_metodo_distribuicoes.png`` — small multiples: histograma + CONAMA + flags.

CSV: ``data/interim/melhor_metodo_por_variavel.csv``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from projeto_pcj import load_semae  # noqa: E402
from projeto_pcj.schema import CONAMA_357_CLASSE2, FEATURES  # noqa: E402

LOG = logging.getLogger("melhor_metodo")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_CSV = PROJECT_ROOT / "data" / "interim" / "melhor_metodo_por_variavel.csv"

MIN_OBS = 30  # mínimo de observações não-NaN para confiar nas métricas
EXPECTED_RATE = 0.07

METODOS = ["IQR_1.5", "IQR_3.0", "Z-score", "MAD", "Percentil_1-99", "Tukey_adj"]

CORES_METODO = {
    "IQR_1.5": "#3498db",
    "IQR_3.0": "#2980b9",
    "Z-score": "#27ae60",
    "MAD": "#e67e22",
    "Percentil_1-99": "#9b59b6",
    "Tukey_adj": "#c0392b",
}


# --- Métodos univariados ------------------------------------------------------


def _flag_iqr(x: np.ndarray, k: float) -> np.ndarray:
    valid = ~np.isnan(x)
    if valid.sum() < MIN_OBS:
        return np.zeros_like(x, dtype=bool)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(x, dtype=bool)
    return ((x < q1 - k * iqr) | (x > q3 + k * iqr)) & valid


def flag_iqr15(x): return _flag_iqr(x, 1.5)
def flag_iqr30(x): return _flag_iqr(x, 3.0)


def flag_zscore(x: np.ndarray, t: float = 3.0) -> np.ndarray:
    valid = ~np.isnan(x)
    if valid.sum() < MIN_OBS:
        return np.zeros_like(x, dtype=bool)
    mu, sigma = np.nanmean(x), np.nanstd(x)
    if sigma == 0:
        return np.zeros_like(x, dtype=bool)
    return (np.abs((x - mu) / sigma) > t) & valid


def flag_mad(x: np.ndarray, t: float = 3.5) -> np.ndarray:
    valid = ~np.isnan(x)
    if valid.sum() < MIN_OBS:
        return np.zeros_like(x, dtype=bool)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x, dtype=bool)
    return (np.abs(0.6745 * (x - med) / mad) > t) & valid


def flag_percentil(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    valid = ~np.isnan(x)
    if valid.sum() < MIN_OBS:
        return np.zeros_like(x, dtype=bool)
    lo, hi = np.nanpercentile(x, [low, high])
    return ((x < lo) | (x > hi)) & valid


def _medcouple(x: np.ndarray) -> float:
    """Medcouple de Brys et al. (2004), implementação O(n²) — ok para n ≤ 1000."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return 0.0
    m = float(np.median(x))
    upper = x[x >= m]
    lower = x[x <= m]
    hs = []
    for xi in upper:
        for xj in lower:
            if xi == m and xj == m:
                continue
            d = xi - xj
            if d == 0:
                continue
            hs.append(((xi - m) - (m - xj)) / d)
    return float(np.median(hs)) if hs else 0.0


def flag_tukey_adj(x: np.ndarray) -> np.ndarray:
    """Boxplot ajustado por assimetria (Hubert & Vandervieren, 2008)."""
    valid = ~np.isnan(x)
    if valid.sum() < MIN_OBS:
        return np.zeros_like(x, dtype=bool)
    mc = _medcouple(x)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(x, dtype=bool)
    if mc >= 0:
        lo = q1 - 1.5 * np.exp(-4 * mc) * iqr
        hi = q3 + 1.5 * np.exp(3 * mc) * iqr
    else:
        lo = q1 - 1.5 * np.exp(-3 * mc) * iqr
        hi = q3 + 1.5 * np.exp(4 * mc) * iqr
    return ((x < lo) | (x > hi)) & valid


METHOD_FN = {
    "IQR_1.5": flag_iqr15,
    "IQR_3.0": flag_iqr30,
    "Z-score": flag_zscore,
    "MAD": flag_mad,
    "Percentil_1-99": flag_percentil,
    "Tukey_adj": flag_tukey_adj,
}


# --- Ground truth via CONAMA --------------------------------------------------


def conama_violation(x: np.ndarray, var: str) -> np.ndarray | None:
    """Marca violação dos limites CONAMA 357/2005 Classe 2 para a variável.

    Retorna ``None`` se a variável não tem limite no schema (não há ground truth).
    """
    limites = CONAMA_357_CLASSE2.get(var)
    if not limites:
        return None
    lo, hi = limites
    out = np.zeros_like(x, dtype=bool)
    valid = ~np.isnan(x)
    if lo is not None:
        out |= valid & (x < lo)
    if hi is not None:
        out |= valid & (x > hi)
    return out


# --- Métricas -----------------------------------------------------------------


def f1_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


# --- Pipeline -----------------------------------------------------------------


def avaliar_variavel(x: np.ndarray, var: str) -> dict:
    """Avalia os 6 métodos para uma variável; retorna dict de resultados."""
    flags = {m: fn(x) for m, fn in METHOD_FN.items()}
    n_valid = int((~np.isnan(x)).sum())

    res: dict[str, dict] = {"variavel": var, "n_obs": n_valid, "criterio": None,
                            "vencedor": None, "score_vencedor": np.nan,
                            "metodos": {}}

    if n_valid < MIN_OBS:
        res["criterio"] = "INSUFICIENTE"
        for m in METODOS:
            res["metodos"][m] = {"n_flags": int(flags[m].sum()), "score": np.nan,
                                 "precision": np.nan, "recall": np.nan, "f1": np.nan,
                                 "jaccard_medio": np.nan}
        return res

    gt = conama_violation(x, var)
    if gt is not None and gt.sum() >= 1:
        res["criterio"] = "CONAMA_F1"
        for m, mask in flags.items():
            prec, rec, f1 = f1_metrics(mask, gt)
            res["metodos"][m] = {"n_flags": int(mask.sum()), "score": f1,
                                 "precision": prec, "recall": rec, "f1": f1,
                                 "jaccard_medio": np.nan}
        res["vencedor"] = max(res["metodos"], key=lambda m: res["metodos"][m]["score"])
        res["score_vencedor"] = res["metodos"][res["vencedor"]]["score"]
    else:
        # Sem CONAMA (ou sem violação observada): consenso entre métodos
        res["criterio"] = "JACCARD" if gt is None else "JACCARD (CONAMA sem violação)"
        jac_medio = {}
        for m, mask in flags.items():
            outros = [flags[k] for k in METODOS if k != m]
            jac = float(np.mean([jaccard(mask, o) for o in outros]))
            jac_medio[m] = jac
            res["metodos"][m] = {"n_flags": int(mask.sum()), "score": jac,
                                 "precision": np.nan, "recall": np.nan, "f1": np.nan,
                                 "jaccard_medio": jac}
        res["vencedor"] = max(jac_medio, key=jac_medio.get)
        res["score_vencedor"] = jac_medio[res["vencedor"]]

    return res


# --- Plots --------------------------------------------------------------------


def plot_heatmap(resultados: list[dict], out: Path) -> None:
    """Heatmap variável × método. Valor: F1 (CONAMA) ou Jaccard (consenso)."""
    mat = pd.DataFrame(
        index=[r["variavel"] for r in resultados],
        columns=METODOS,
        dtype=float,
    )
    anot = pd.DataFrame("", index=mat.index, columns=mat.columns)
    for r in resultados:
        if r["criterio"] == "INSUFICIENTE":
            continue
        for m in METODOS:
            mat.loc[r["variavel"], m] = r["metodos"][m]["score"]
            if m == r["vencedor"]:
                anot.loc[r["variavel"], m] = f"{r['metodos'][m]['score']:.2f}*"
            else:
                anot.loc[r["variavel"], m] = f"{r['metodos'][m]['score']:.2f}"

    # Marca variáveis com critério Jaccard via tag no nome
    novo_index = []
    for r in resultados:
        tag = "" if r["criterio"] == "CONAMA_F1" else " (J)" if r["criterio"].startswith("JACCARD") else " (×)"
        novo_index.append(r["variavel"] + tag)
    mat.index = novo_index
    anot.index = novo_index

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(mat.astype(float), annot=anot.values, fmt="", cmap="viridis",
                vmin=0, vmax=1, cbar_kws={"label": "Score (F1 ou Jaccard)"},
                linewidths=0.3, ax=ax)
    ax.set_title("Melhor método por variável — score por método (* = vencedor da variável)\n"
                 "Sufixo (J) = critério Jaccard (sem CONAMA); (×) = obs insuficientes")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_ranking(resultados: list[dict], out: Path) -> None:
    """Barras horizontais com método-vencedor e score por variável."""
    dados = [(r["variavel"], r["vencedor"] or "—", r["score_vencedor"], r["criterio"])
             for r in resultados]
    dados = sorted([d for d in dados if d[1] != "—"], key=lambda d: d[2], reverse=True)
    df = pd.DataFrame(dados, columns=["variavel", "vencedor", "score", "criterio"])

    fig, ax = plt.subplots(figsize=(10, 8))
    cores = [CORES_METODO.get(v, "grey") for v in df["vencedor"]]
    ax.barh(df["variavel"], df["score"], color=cores)
    for i, (v, c, s) in enumerate(zip(df["variavel"], df["vencedor"], df["score"], strict=False)):
        ax.text(s + 0.01, i, f"{c} ({s:.2f})", va="center", fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Score do método vencedor (F1 ou Jaccard)")
    ax.set_title("Método vencedor por variável (sem imputação)")
    ax.invert_yaxis()
    # Legenda
    from matplotlib.patches import Patch
    legenda = [Patch(facecolor=c, label=m) for m, c in CORES_METODO.items()]
    ax.legend(handles=legenda, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


def plot_distribuicoes(df_uni: pd.DataFrame, resultados: list[dict], out: Path) -> None:
    """Small multiples: histograma + limite CONAMA + rugplot de flags por método."""
    elegiveis = [r for r in resultados if r["criterio"] != "INSUFICIENTE"]
    n = len(elegiveis)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for i, r in enumerate(elegiveis):
        ax = axes[i]
        var = r["variavel"]
        x = df_uni[var].to_numpy()
        x_clean = x[~np.isnan(x)]
        if len(x_clean) == 0:
            ax.set_visible(False)
            continue

        # Histograma
        ax.hist(x_clean, bins=30, color="#bdc3c7", edgecolor="white")
        ax.set_title(f"{var}\nvencedor: {r['vencedor']} ({r['score_vencedor']:.2f})", fontsize=9)

        # Linhas CONAMA
        lim = CONAMA_357_CLASSE2.get(var)
        if lim:
            lo, hi = lim
            if lo is not None:
                ax.axvline(lo, color="red", lw=1.5, ls="--", label=f"CONAMA min={lo}")
            if hi is not None:
                ax.axvline(hi, color="red", lw=1.5, ls="--", label=f"CONAMA max={hi}")

        # Rugplot por método (offset vertical para separar visualmente)
        ylim = ax.get_ylim()
        passo = (ylim[1] - ylim[0]) / (len(METODOS) + 2)
        y_base = ylim[1] * 1.02
        for j, m in enumerate(METODOS):
            flags_m = METHOD_FN[m](x)
            xs = x[flags_m & ~np.isnan(x)]
            if len(xs):
                ax.scatter(xs, np.full_like(xs, y_base + j * passo * 0.5),
                           marker="|", s=80, color=CORES_METODO[m],
                           linewidths=2, label=m if i == 0 else None)
        ax.set_ylim(ylim[0], y_base + len(METODOS) * passo * 0.5)
        ax.tick_params(axis="both", labelsize=8)

    # Legenda única na primeira subplot
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))

    # Esconde subplots vazios
    for k in range(len(elegiveis), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Distribuição por variável — flags de cada método na régua superior", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Salvo: %s", out.name)


# --- Main ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = load_semae()
    monthly = df[(df["Mes"] != "Ano") & (df["Calc"] == "Méd.")].reset_index(drop=True)
    LOG.info("Observações: %d", len(monthly))

    resultados: list[dict] = []
    for var in FEATURES:
        r = avaliar_variavel(monthly[var].to_numpy(), var)
        resultados.append(r)
        LOG.info("%-25s n=%3d  critério=%-30s vencedor=%s score=%.3f",
                 var, r["n_obs"], r["criterio"], r["vencedor"] or "—", r["score_vencedor"] or 0.0)

    # CSV
    linhas = []
    for r in resultados:
        for m in METODOS:
            d = r["metodos"][m]
            linhas.append({
                "variavel": r["variavel"], "n_obs": r["n_obs"], "metodo": m,
                "criterio": r["criterio"], "vencedor": (m == r["vencedor"]),
                "score": d["score"], "n_flags": d["n_flags"],
                "precision": d["precision"], "recall": d["recall"], "f1": d["f1"],
                "jaccard_medio": d["jaccard_medio"],
            })
    pd.DataFrame(linhas).to_csv(OUT_CSV, index=False, encoding="utf-8")
    LOG.info("CSV salvo em %s", OUT_CSV)

    # Plots
    plot_heatmap(resultados, FIG_DIR / "melhor_metodo_heatmap.png")
    plot_ranking(resultados, FIG_DIR / "melhor_metodo_ranking.png")
    plot_distribuicoes(monthly[FEATURES], resultados,
                       FIG_DIR / "melhor_metodo_distribuicoes.png")


if __name__ == "__main__":
    main()
