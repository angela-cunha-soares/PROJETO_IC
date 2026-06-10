"""Figura dos outliers confirmados (consenso de métodos + violação CONAMA).

Lê ``data/interim/outliers_confirmados.csv`` e gera duas visões em
``reports/figures/``:

1. ``outliers_confirmados_heatmap.png`` — meses confirmados (linhas) × parâmetros
   CONAMA (colunas); a célula é destacada quando o valor viola o limite legal e
   anotada com o valor real medido. Mostra QUAIS parâmetros disparam cada outlier.
2. ``outliers_confirmados_timeline.png`` — linha do tempo (2009–2024) com o nº de
   métodos e violações por mês confirmado, com a crise do Cantareira sombreada.

Uso:
    python scripts/plot_outliers_confirmados.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matplotlib.patches import Patch, Rectangle  # noqa: E402

from features.build_features import (  # noqa: E402
    PARAMS_INDICATIVOS, tabela_modelagem, violacoes_conama,
)
from projeto_pcj.load import load_semae  # noqa: E402
from projeto_pcj.schema import CONAMA_357_CLASSE2, FEATURES  # noqa: E402

LOG = logging.getLogger("plot_outliers_confirmados")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
CONF_CSV = PROJECT_ROOT / "data" / "interim" / "outliers_confirmados.csv"
CRISE = (2014, 2016)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not CONF_CSV.is_file():
        raise FileNotFoundError(
            f"{CONF_CSV} não encontrado — rode antes: "
            "python scripts/tabela_outliers_por_metodo.py"
        )

    conf = pd.read_csv(CONF_CSV)
    chaves = set(zip(conf["Ano"], conf["Mes"]))

    # Recupera os valores completos (todas as variáveis) dos meses confirmados.
    m = tabela_modelagem(load_semae())
    sel = m[[(a, me) in chaves for a, me in zip(m["Ano"], m["Mes"])]].copy()
    sel["Mes"] = sel["Mes"].astype(str)
    extra = ["n_metodos", "n_violacoes_conama", "n_violacoes_sem_indicativos"]
    sel = sel.merge(conf[["Ano", "Mes"] + extra], on=["Ano", "Mes"])
    sel = sel.sort_values(["Ano", "mes_num"]).reset_index(drop=True)

    norm_cols = [c for c in FEATURES if c in CONAMA_357_CLASSE2]
    viol = violacoes_conama(sel, norm_cols)

    # Rótulo mostra nº de métodos e nº de violações DESCONTANDO indicativos (sem Fe).
    rotulos = [f"{r.Ano}-{r.Mes}  ({int(r.n_metodos)}m · "
               f"{int(r.n_violacoes_sem_indicativos)} viol. s/ Fe)"
               for r in sel.itertuples()]

    # --- Figura 1: heatmap de violações × valor ---------------------------
    fig, ax = plt.subplots(figsize=(max(9, 0.65 * len(norm_cols)),
                                    0.5 * len(sel) + 2.5))
    M = viol[norm_cols].astype(int).to_numpy()
    # Violações de parâmetros indicativos (Fe) não pintam de vermelho — viram hachura.
    M_disp = M.copy()
    for j, col in enumerate(norm_cols):
        if col in PARAMS_INDICATIVOS:
            M_disp[:, j] = 0
    ax.imshow(M_disp, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(norm_cols)))
    ax.set_xticklabels(norm_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels(rotulos, fontsize=8)
    # Anota o valor medido; negrito quando viola. Fe violado recebe hachura laranja.
    for i in range(len(sel)):
        for j, col in enumerate(norm_cols):
            indicativo = col in PARAMS_INDICATIVOS
            val = sel.iloc[i][col]
            viola = bool(M[i, j])
            if viola and indicativo:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="////", edgecolor="#e67e22", lw=0.5))
            if pd.isna(val):
                txt, w, cor = "–", "normal", "#888"
            else:
                txt = f"{val:.2f}" if val < 100 else f"{val:.0f}"
                w = "bold" if viola else "normal"
                cor = "#9c4500" if (viola and indicativo) else ("black" if viola else "#888")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    fontweight=w, color=cor)
    ax.set_title("Outliers confirmados — parâmetros que violam a CONAMA 357/2005 Cl.2\n"
                 "(vermelho = viola; hachura laranja = violação só indicativa)", fontsize=11)
    ax.legend(handles=[
        Patch(facecolor="#c0392b", label="viola (limite confiável)"),
        Patch(facecolor="white", edgecolor="#e67e22", hatch="////",
              label="violação indicativa (Fe total vs dissolvido)"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.text(
        0.5, -0.02,
        "Ressalva: a CONAMA 357/2005 limita o Fe DISSOLVIDO (0,3 mg/L), mas o SEMAE "
        "mede Fe TOTAL → a violação de Fe é só indicativa (hachurada) e não conta no "
        "rótulo das linhas.\nMn (0,1) é limite de Mn total e coincide com a medição.",
        ha="center", va="top", fontsize=7.5, style="italic", color="#555",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "outliers_confirmados_heatmap.png", dpi=120, bbox_inches="tight")
    LOG.info("Salvo outliers_confirmados_heatmap.png")

    # --- Figura 3: violações com Fe vs sem Fe por mês ---------------------
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(sel) + 1.5))
    y = np.arange(len(sel))
    rotulo_mes = [f"{r.Ano}-{r.Mes}" for r in sel.itertuples()]
    ax.barh(y + 0.2, sel["n_violacoes_conama"], height=0.4,
            color="#c0392b", label="com Fe (norma literal)")
    ax.barh(y - 0.2, sel["n_violacoes_sem_indicativos"], height=0.4,
            color="#2980b9", label="sem Fe (de-enviesado)")
    ax.set_yticks(y)
    ax.set_yticklabels(rotulo_mes, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("nº de parâmetros CONAMA violados")
    ax.set_title("Outliers confirmados: violações com vs. sem o Ferro (indicativo)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "outliers_confirmados_com_sem_fe.png", dpi=120, bbox_inches="tight")
    LOG.info("Salvo outliers_confirmados_com_sem_fe.png")

    # --- Figura 2: linha do tempo -----------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axvspan(CRISE[0] - 0.5, CRISE[1] + 0.5, color="red", alpha=0.10,
               label=f"Crise Cantareira ({CRISE[0]}–{CRISE[1]})")
    x = sel["Ano"] + (sel["mes_num"] - 1) / 12
    ax.scatter(x, sel["n_metodos"], s=80, c="#c0392b",
               edgecolor="k", zorder=3, label="nº de métodos (de 6)")
    for r, xi in zip(sel.itertuples(), x):
        ax.annotate(r.Mes, (xi, r.n_metodos), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xlim(2008.5, 2024.5)
    ax.set_ylim(3.5, 6.5)
    ax.set_xlabel("ano")
    ax.set_ylabel("nº de métodos que marcaram")
    ax.set_title("Outliers confirmados ao longo do tempo")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "outliers_confirmados_timeline.png", dpi=120, bbox_inches="tight")
    LOG.info("Salvo outliers_confirmados_timeline.png")


if __name__ == "__main__":
    main()
