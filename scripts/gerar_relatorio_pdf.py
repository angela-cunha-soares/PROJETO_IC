"""Gera o relatório final em PDF (projeto §6: "PDFs de resultados de validação").

Monta um PDF multipágina a partir das tabelas (``reports/tables/``) e figuras
(``reports/figures/``) já produzidas por ``run_train.py`` / ``run_evaluate.py``.
Usa ``matplotlib.backends.backend_pdf`` — sem dependência extra.

Pré-requisitos: rodar antes ``run_preprocess`` → ``run_train`` → ``run_evaluate``
(e, opcionalmente, ``run_outliers`` e ``calibrar_contamination``).

Saída: ``reports/relatorio_final.pdf``.

Uso:
    python scripts/gerar_relatorio_pdf.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402

LOG = logging.getLogger("gerar_relatorio_pdf")
OUT_PDF = config.REPORTS_DIR / "relatorio_final.pdf"


def _tabela(nome: str) -> pd.DataFrame | None:
    p = config.TABLES_DIR / nome
    return pd.read_csv(p) if p.is_file() else None


def _pagina_texto(pdf: PdfPages, titulo: str, linhas: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 retrato
    fig.text(0.08, 0.94, titulo, fontsize=18, fontweight="bold")
    fig.text(0.08, 0.90, "\n".join(linhas), fontsize=11, va="top", family="monospace")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _pagina_tabela(pdf: PdfPages, titulo: str, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(titulo, fontsize=14, fontweight="bold", loc="left")
    t = ax.table(cellText=df.round(4).values, colLabels=df.columns,
                 loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    pdf.savefig(fig)
    plt.close(fig)


def _pagina_imagem(pdf: PdfPages, titulo: str, img_path: Path) -> bool:
    if not img_path.is_file():
        return False
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
    ax.imshow(mpimg.imread(img_path))
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)
    return True


def _resumo_linhas() -> list[str]:
    linhas = ["Aplicação de Machine Learning — Qualidade da água na bacia do PCJ",
              "SEMAE Piracicaba, 2009–2024 · base de modelagem: 192 meses (Méd.)", ""]
    km = _tabela("kmeans_varredura.csv")
    if km is not None:
        melhor = km.loc[km["silhueta"].idxmax()]
        linhas.append(f"K-Means : melhor k={int(melhor['k'])}, "
                      f"silhueta={melhor['silhueta']:.3f}")
    rf = _tabela("rf_metricas.csv")
    if rf is not None:
        m = rf.iloc[0]
        linhas.append(f"Random Forest : acurácia={m['acuracia']:.3f}, "
                      f"F1={m['f1']:.3f}, precisão={m['precisao']:.3f}, "
                      f"recall={m['recall']:.3f}")
    cal = _tabela("contamination_calibracao.csv")
    if cal is not None:
        best = cal.sort_values("estabilidade_jaccard", ascending=False).iloc[0]
        linhas.append(f"Isolation Forest : contamination calibrado="
                      f"{best['contamination']:.2f} "
                      f"(estabilidade={best['estabilidade_jaccard']:.3f})")
    imp = _tabela("validacao_imputacao_kmeans.csv")
    if imp is not None:
        cc = imp[imp["conjunto"] == "complete-case"]["silhueta"].iloc[0]
        ip = imp[imp["conjunto"] == "imputado"]["silhueta"].iloc[0]
        linhas += ["", f"Validação da imputação (silhueta K-Means): "
                       f"complete-case={cc:.3f} vs imputado={ip:.3f}"]
    return linhas


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config.ensure_dirs()
    fig_dir = config.FIGURES_DIR

    with PdfPages(OUT_PDF) as pdf:
        _pagina_texto(pdf, "Relatório final — Resultados", _resumo_linhas())

        cm = _tabela("rf_matriz_confusao.csv")
        if cm is not None:
            _pagina_tabela(pdf, "Random Forest — Matriz de confusão", cm)
        imp = _tabela("validacao_imputacao_kmeans.csv")
        if imp is not None:
            _pagina_tabela(pdf, "Validação da imputação — impacto na silhueta", imp)

        figuras = [
            ("K-Means — cotovelo e silhueta", "eval_kmeans_cotovelo_silhueta.png"),
            ("Random Forest — matriz de confusão", "eval_rf_matriz_confusao.png"),
            ("Random Forest — importância das variáveis", "eval_rf_importancias.png"),
            ("Isolation Forest — anomalias", "eval_iforest_scatter.png"),
            ("Calibração de contamination (CV)", "contamination_calibracao.png"),
            ("Outliers confirmados — violações CONAMA", "outliers_confirmados_heatmap.png"),
            ("Correlação entre variáveis", "eval_correlacao.png"),
        ]
        incluidas = 0
        for titulo, nome in figuras:
            if _pagina_imagem(pdf, titulo, fig_dir / nome):
                incluidas += 1

    LOG.info("PDF gerado: %s (%d páginas de figura)", OUT_PDF, incluidas)


if __name__ == "__main__":
    main()
