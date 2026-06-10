"""Pipeline de avaliação: gera figuras de validação e o relatório consolidado.

Pré-requisito: rodar ``run_preprocess.py`` e ``run_train.py`` antes.

Saídas em ``reports/figures/``:
    * ``eval_kmeans_cotovelo_silhueta.png``, ``eval_kmeans_scatter.png``
    * ``eval_rf_matriz_confusao.png``, ``eval_rf_importancias.png``
    * ``eval_iforest_scatter.png``, ``eval_correlacao.png``
E em ``reports/``:
    * ``relatorio_resultados.md`` — síntese das métricas.

Uso:
    python scripts/run_evaluate.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from visualization import plots  # noqa: E402

LOG = logging.getLogger("run_evaluate")
ID_COLS = ["data", "Ano", "Mes", "mes_num", "trimestre", "estacao",
           "cluster", "inadequada", "severidade_alta", "n_violacoes",
           "anomalia", "anomaly_score"]
FIG = config.FIGURES_DIR


def _ler(nome: str) -> pd.DataFrame | None:
    path = config.TABLES_DIR / nome
    return pd.read_csv(path) if path.is_file() else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config.ensure_dirs()

    rotulados = pd.read_csv(config.PROCESSED_DIR / "dados_rotulados.csv")
    features = [c for c in rotulados.columns if c not in ID_COLS]
    secoes: dict[str, str] = {}

    # Correlação geral
    plots.heatmap_correlacao(rotulados, features,
                             salvar_em=FIG / "eval_correlacao.png")

    # K-Means
    varredura = _ler("kmeans_varredura.csv")
    if varredura is not None:
        plots.curva_cotovelo_silhueta(
            varredura, salvar_em=FIG / "eval_kmeans_cotovelo_silhueta.png")
        if {"pH", "TURB.(FTU)"}.issubset(rotulados.columns) and "cluster" in rotulados:
            plots.scatter_clusters(rotulados, "pH", "TURB.(FTU)",
                                   rotulados["cluster"],
                                   salvar_em=FIG / "eval_kmeans_scatter.png")
        melhor = varredura.loc[varredura["silhueta"].idxmax()]
        secoes["K-Means"] = (
            f"- k escolhido (silhueta): **{int(melhor['k'])}**\n"
            f"- silhueta máxima: **{melhor['silhueta']:.3f}** "
            f"({'efetivo' if melhor['silhueta'] > config.SILHOUETTE_GOOD else 'fraco'}, limiar 0,5)"
        )

    # Random Forest
    rf_met = _ler("rf_metricas.csv")
    cm = _ler("rf_matriz_confusao.csv")
    imp = _ler("rf_importancias.csv")
    if rf_met is not None and cm is not None:
        cm_idx = cm.set_index(cm.columns[0])
        plots.matriz_confusao(cm_idx, salvar_em=FIG / "eval_rf_matriz_confusao.png")
        if imp is not None:
            serie = imp.set_index("variavel")["importancia"]
            plots.importancia_variaveis(serie, salvar_em=FIG / "eval_rf_importancias.png")
        m = rf_met.iloc[0].to_dict()
        secoes["Random Forest"] = (
            f"- acurácia: **{m['acuracia']:.3f}** | F1: **{m['f1']:.3f}** "
            f"| precisão: {m['precisao']:.3f} | recall: {m['recall']:.3f}"
        )

    # Isolation Forest
    if "anomalia" in rotulados.columns:
        flags = rotulados["anomalia"].astype(bool)
        if {"P(ppm P)", "TURB.(FTU)"}.issubset(rotulados.columns):
            plots.scatter_anomalias(rotulados, "TURB.(FTU)", "pH", flags,
                                    salvar_em=FIG / "eval_iforest_scatter.png")
        elif {"Cond.(uS/cm)", "pH"}.issubset(rotulados.columns):
            plots.scatter_anomalias(rotulados, "Cond.(uS/cm)", "pH", flags,
                                    salvar_em=FIG / "eval_iforest_scatter.png")
        secoes["Isolation Forest"] = (
            f"- taxa de anomalias: **{flags.mean():.3f}** "
            f"({int(flags.sum())} de {len(flags)} pontos)"
        )

    from models.reports import resumo_markdown

    md = resumo_markdown(secoes)
    (config.REPORTS_DIR / "relatorio_resultados.md").write_text(md, encoding="utf-8")
    LOG.info("Relatório salvo em reports/relatorio_resultados.md")
    LOG.info("Figuras de avaliação geradas em %s", FIG)

    # Monta o relatório final em PDF (projeto §6) a partir das tabelas/figuras.
    import subprocess

    res = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("gerar_relatorio_pdf.py"))],
        cwd=config.PROJECT_ROOT,
    )
    if res.returncode == 0:
        LOG.info("Relatório PDF gerado em reports/relatorio_final.pdf")
    else:
        LOG.warning("Falha ao gerar o PDF (código %d)", res.returncode)


if __name__ == "__main__":
    main()
