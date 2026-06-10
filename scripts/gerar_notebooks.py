"""Gera os notebooks 00–10 (formato nbformat v4) a partir de células declaradas aqui.

Mantém os notebooks finos: eles importam de ``src/`` e documentam cada etapa,
sem duplicar lógica. Rode após qualquer mudança na API de ``src/``:

    python scripts/gerar_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = PROJECT_ROOT / "notebooks"

BOOT = (
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n"
    "import matplotlib.pyplot as plt"
)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.splitlines(keepends=True),
    }


# (arquivo, título, [células])
NOTEBOOKS: list[tuple[str, list[dict]]] = [
    ("00_visao_geral.ipynb", [
        md("# 00 · Visão geral\n\n"
           "Projeto de IC (FAPESP) — **Machine Learning para qualidade da água na bacia do PCJ**.\n"
           "Fonte: SEMAE Piracicaba (água bruta), 2009–2024.\n\n"
           "Pipeline: carga → faltantes/imputação → padronização → EDA → outliers → "
           "K-Means → Random Forest → Isolation Forest → validação → relatório.\n\n"
           "Os notebooks 01–10 reproduzem cada etapa chamando funções de `src/`. "
           "Veja `docs/metodologia.md` e `docs/decisoes_de_projeto.md`."),
        code(BOOT + "\nfrom projeto_pcj.load import load_semae\n"
             "df = load_semae()\n"
             "print('linhas:', len(df))\n"
             "df.head()"),
    ]),
    ("01_carga_e_inspecao.ipynb", [
        md("# 01 · Carga e inspeção\n\nParser numérico BR, tipagem canônica e schema."),
        code(BOOT + "\nfrom projeto_pcj.load import load_semae, faltantes_por_variavel\n"
             "from projeto_pcj.schema import FEATURES\n"
             "df = load_semae()\n"
             "df.info()"),
        code("df[['Ano','Mes','Calc']+FEATURES[:6]].head(9)"),
        code("df.describe().T"),
    ]),
    ("02_pre_processamento.ipynb", [
        md("# 02 · Pré-processamento\n\nTabela de modelagem (Méd. mensal), faltantes e teste MCAR (Little)."),
        code(BOOT + "\nfrom projeto_pcj.load import load_semae\n"
             "from projeto_pcj.schema import FEATURES\n"
             "from features.build_features import tabela_modelagem\n"
             "from preprocessing.missing import resumo_faltantes, little_mcar_test\n"
             "m = tabela_modelagem(load_semae())\n"
             "print('amostras:', len(m))"),
        code("resumo_faltantes(m, FEATURES)"),
        md("Teste de Little: `p < 0,05` rejeita MCAR (faltantes não totalmente aleatórios)."),
        code("little_mcar_test(m, FEATURES)"),
    ]),
    ("03_imputacao.ipynb", [
        md("# 03 · Imputação\n\nRegras da ADR-004: excluir >30%; KNN 5–30%; média/mediana <5%."),
        code(BOOT + "\nfrom projeto_pcj.load import load_semae\n"
             "from projeto_pcj.schema import FEATURES\n"
             "from features.build_features import tabela_modelagem\n"
             "from preprocessing.missing import planejar_imputacao, imputar, validar_imputacao\n"
             "m = tabela_modelagem(load_semae())\n"
             "plano = planejar_imputacao(m, FEATURES)\n"
             "print('excluir:', plano.excluir)\nprint('knn:', plano.knn)"),
        code("imp, plano = imputar(m, FEATURES)\n"
             "validar_imputacao(m, imp, plano.manter)"),
    ]),
    ("04_analise_exploratoria.ipynb", [
        md("# 04 · Análise exploratória\n\nDistribuições, correlação e série temporal."),
        code(BOOT + "\nfrom projeto_pcj.load import load_semae\n"
             "from features.build_features import tabela_modelagem\n"
             "from preprocessing.missing import imputar, planejar_imputacao\n"
             "from projeto_pcj.schema import FEATURES\n"
             "from visualization import plots\n"
             "m = tabela_modelagem(load_semae())\n"
             "imp, plano = imputar(m, FEATURES)\n"
             "feats = plano.manter"),
        code("plots.histograma(imp, 'TURB.(FTU)'); plt.show()"),
        code("plots.heatmap_correlacao(imp, feats); plt.show()"),
        code("plots.serie_temporal(imp, 'Cond.(uS/cm)'); plt.show()"),
    ]),
    ("05_outliers_iqr_vs_ensemble.ipynb", [
        md("# 05 · Outliers — IQR vs. ensemble\n\n"
           "Baseline IQR (Tukey) vs. votação IForest+LOF+KNN. Scripts detalhados: "
           "`comparar_metodos_outliers.py` e `melhor_metodo_por_variavel.py`."),
        code(BOOT + "\nfrom projeto_pcj.load import load_semae\n"
             "from features.build_features import tabela_modelagem\n"
             "from preprocessing.missing import imputar, planejar_imputacao\n"
             "from preprocessing.scaling import matriz_scaled\n"
             "from preprocessing.outliers import iqr_outliers_tabela, ensemble_outliers, jaccard\n"
             "from projeto_pcj.schema import FEATURES\n"
             "m = tabela_modelagem(load_semae())\n"
             "imp, plano = imputar(m, FEATURES); feats = plano.manter\n"
             "X = matriz_scaled(imp, feats)"),
        code("iqr_outliers_tabela(imp, feats)"),
        code("res = ensemble_outliers(X)\n"
             "print('outliers (ensemble):', int(res.outliers.sum()))\n"
             "res.por_metodo.sum()"),
        md("### Outlier confirmado (sem ground truth)\n\n"
           "Critério: marcado por ≥4 dos 6 métodos **e** viola ≥1 limite CONAMA 357/2005.\n"
           "Tabelas e figuras geradas por scripts dedicados:\n\n"
           "```bash\n"
           "python scripts/tabela_outliers_por_metodo.py   # data/interim/outliers_confirmados.csv\n"
           "python scripts/plot_outliers_por_metodo.py      # painel PCA-2D por método\n"
           "python scripts/plot_outliers_confirmados.py     # heatmap mês × parâmetro CONAMA\n"
           "```"),
        code("import pandas as pd, config\n"
             "conf = config.INTERIM_DIR / 'outliers_confirmados.csv'\n"
             "pd.read_csv(conf) if conf.is_file() else 'rode tabela_outliers_por_metodo.py'"),
    ]),
    ("06_kmeans_clustering.ipynb", [
        md("# 06 · K-Means\n\nCotovelo + silhueta, ajuste e perfil dos clusters."),
        code(BOOT + "\nimport pandas as pd\nimport config\n"
             "from models import kmeans as km\n"
             "scaled = pd.read_csv(config.PROCESSED_DIR / 'dados_scaled.csv')\n"
             "imp = pd.read_csv(config.PROCESSED_CSV)\n"
             "ID = ['data','Ano','Mes','mes_num','trimestre','estacao']\n"
             "feats = [c for c in scaled.columns if c not in ID]\n"
             "X = scaled[feats].to_numpy(float)"),
        code("v = km.varrer_k(X)\nfrom visualization import plots\n"
             "plots.curva_cotovelo_silhueta(v.tabela); plt.show()\n"
             "print('k silhueta:', v.k_silhueta, '| k cotovelo:', v.k_cotovelo)"),
        code("modelo = km.ajustar(X, v.k_silhueta)\n"
             "km.perfil_clusters(imp, modelo.labels_, feats)"),
    ]),
    ("07_random_forest.ipynb", [
        md("# 07 · Random Forest\n\nRotulagem CONAMA → **severidade** (alvo balanceado), "
           "anti-leakage, métricas e importância de variáveis."),
        code(BOOT + "\nimport pandas as pd\nimport config\n"
             "from features.build_features import rotular_severidade, rotular_conama, features_sem_vazamento, contar_violacoes\n"
             "from models import random_forest as rf\n"
             "scaled = pd.read_csv(config.PROCESSED_DIR / 'dados_scaled.csv')\n"
             "imp = pd.read_csv(config.PROCESSED_CSV)\n"
             "ID = ['data','Ano','Mes','mes_num','trimestre','estacao']\n"
             "feats = [c for c in scaled.columns if c not in ID]"),
        code("print('rótulo estrito (degenerado):', rotular_conama(imp)[0].value_counts().to_dict())\n"
             "y, usadas = rotular_severidade(imp)\n"
             "print('severidade:', y.value_counts().to_dict())\n"
             "Xfeats = features_sem_vazamento(feats, usadas)\n"
             "print('features RF (sem vazamento):', Xfeats)"),
        code("res = rf.treinar(scaled[Xfeats], y)\n"
             "print(res.metricas, '| CV-F1:', round(res.cv_f1.mean(),3))\n"
             "from visualization import plots\n"
             "plots.matriz_confusao(res.matriz_confusao); plt.show()\n"
             "plots.importancia_variaveis(res.importancias); plt.show()"),
    ]),
    ("08_isolation_forest.ipynb", [
        md("# 08 · Isolation Forest\n\nDetecção de anomalias / picos de contaminação."),
        code(BOOT + "\nimport pandas as pd\nimport config\n"
             "from models import isolation_forest as iso\n"
             "from visualization import plots\n"
             "scaled = pd.read_csv(config.PROCESSED_DIR / 'dados_scaled.csv')\n"
             "imp = pd.read_csv(config.PROCESSED_CSV)\n"
             "ID = ['data','Ano','Mes','mes_num','trimestre','estacao']\n"
             "feats = [c for c in scaled.columns if c not in ID]\n"
             "X = scaled[feats].to_numpy(float)"),
        code("res = iso.detectar(X)\nprint('taxa de anomalias:', res.taxa)\n"
             "iso.anomalias_detalhadas(imp, res, feats).head(10)"),
        code("plots.scatter_anomalias(imp, 'Cond.(uS/cm)', 'pH', res.flags); plt.show()"),
    ]),
    ("09_validacao_consolidada.ipynb", [
        md("# 09 · Validação consolidada\n\nLê as tabelas de `reports/tables/` geradas pelos scripts."),
        code(BOOT + "\nimport pandas as pd\nimport config\n"
             "def ler(n):\n"
             "    p = config.TABLES_DIR / n\n"
             "    return pd.read_csv(p) if p.is_file() else None\n"
             "print('Rode antes: run_preprocess.py → run_train.py → run_evaluate.py')"),
        code("ler('kmeans_varredura.csv')"),
        code("ler('rf_metricas.csv')"),
        code("ler('rf_importancias.csv')"),
        code("ler('iforest_anomalias.csv')"),
    ]),
    ("10_relatorio_final.ipynb", [
        md("# 10 · Relatório final\n\nSíntese executiva. O relatório textual é gerado em "
           "`reports/relatorio_resultados.md` por `run_evaluate.py`."),
        code(BOOT + "\nimport config\n"
             "rel = config.REPORTS_DIR / 'relatorio_resultados.md'\n"
             "print(rel.read_text(encoding='utf-8') if rel.is_file() else 'Rode run_evaluate.py primeiro.')"),
        md("### Figuras\nVer `reports/figures/` (`eval_*.png`, `missing_*.png`, "
           "`outliers_*.png`, `metodos_*.png`)."),
    ]),
]


def build(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    for nome, cells in NOTEBOOKS:
        (NB_DIR / nome).write_text(
            json.dumps(build(cells), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print("gerado:", nome)


if __name__ == "__main__":
    main()
