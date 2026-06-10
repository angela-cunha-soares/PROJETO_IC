"""Pipeline de modelagem: K-Means, Random Forest e Isolation Forest.

Pré-requisito: rodar ``scripts/run_preprocess.py`` antes (gera os CSVs em
``data/processed/``).

Saídas:
    * ``models/kmeans.joblib``, ``models/random_forest.joblib``, ``models/isolation_forest.joblib``
    * ``data/processed/dados_rotulados.csv`` — features + cluster + inadequada + anomalia.
    * ``reports/tables/kmeans_varredura.csv``, ``rf_metricas.csv``, ``rf_importancias.csv``,
      ``rf_matriz_confusao.csv``, ``iforest_anomalias.csv``.

Uso:
    python scripts/run_train.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from features.build_features import (  # noqa: E402
    contar_violacoes, features_sem_vazamento, rotular_conama, rotular_severidade,
)
from models import isolation_forest as iso  # noqa: E402
from models import kmeans as km  # noqa: E402
from models import random_forest as rf  # noqa: E402
from models.reports import salvar_modelo, salvar_tabela  # noqa: E402

LOG = logging.getLogger("run_train")
ID_COLS = ["data", "Ano", "Mes", "mes_num", "trimestre", "estacao"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config.ensure_dirs()

    imputado = pd.read_csv(config.PROCESSED_CSV, encoding="utf-8")
    scaled = pd.read_csv(config.PROCESSED_DIR / "dados_scaled.csv", encoding="utf-8")
    features = [c for c in scaled.columns if c not in ID_COLS]
    Xs = scaled[features].to_numpy(dtype=float)
    LOG.info("Features de modelagem (%d): %s", len(features), features)

    saida = imputado.copy()

    # --- K-Means ----------------------------------------------------------
    varredura = km.varrer_k(Xs)
    salvar_tabela(varredura.tabela, "kmeans_varredura")
    k = varredura.k_silhueta
    LOG.info("k (silhueta)=%d | k (cotovelo)=%d → usando k=%d",
             varredura.k_silhueta, varredura.k_cotovelo, k)
    modelo_km = km.ajustar(Xs, k)
    saida["cluster"] = modelo_km.labels_
    salvar_modelo(modelo_km, "kmeans")
    salvar_tabela(km.perfil_clusters(imputado, modelo_km.labels_, features),
                  "kmeans_perfil", index=True)

    # --- Random Forest (rotulagem CONAMA, anti-leakage) -------------------
    # Rótulo estrito (união de violações): degenerado em água bruta — registrado.
    rotulo_estrito, _ = rotular_conama(imputado)
    LOG.info("Rótulo estrito CONAMA (união): %s — em água bruta tende a 1 classe.",
             rotulo_estrito.value_counts().to_dict())
    # Alvo de fato: severidade (nº de violações binarizado pela mediana).
    rotulo, vars_rotulo = rotular_severidade(imputado)
    LOG.info("Violações por amostra: min=%d med=%.1f max=%d",
             int(contar_violacoes(imputado).min()),
             float(contar_violacoes(imputado).median()),
             int(contar_violacoes(imputado).max()))
    LOG.info("Rótulo severidade: %s | variáveis usadas na regra: %s",
             rotulo.value_counts().to_dict(), vars_rotulo)
    feats_rf = features_sem_vazamento(features, vars_rotulo)
    LOG.info("Features do RF (sem vazamento, %d): %s", len(feats_rf), feats_rf)
    if rotulo.nunique() < 2:
        LOG.warning("Rótulo com uma única classe — Random Forest ignorado.")
    else:
        res_rf = rf.treinar(scaled[feats_rf], rotulo,
                            nomes_classes=["baixa severidade", "alta severidade"])
        saida["n_violacoes"] = contar_violacoes(imputado).values
        saida["inadequada"] = rotulo_estrito.values
        saida["severidade_alta"] = rotulo.values
        salvar_modelo(res_rf.modelo, "random_forest")
        salvar_tabela(pd.DataFrame([res_rf.metricas]), "rf_metricas")
        salvar_tabela(res_rf.importancias.rename("importancia").reset_index()
                      .rename(columns={"index": "variavel"}), "rf_importancias")
        salvar_tabela(res_rf.matriz_confusao, "rf_matriz_confusao", index=True)
        LOG.info("RF métricas: %s | CV-F1 média=%.3f",
                 res_rf.metricas, float(res_rf.cv_f1.mean()))

    # --- Isolation Forest -------------------------------------------------
    res_iso = iso.detectar(Xs)
    saida["anomalia"] = res_iso.flags
    saida["anomaly_score"] = res_iso.scores
    salvar_modelo(res_iso.modelo, "isolation_forest")
    salvar_tabela(iso.anomalias_detalhadas(imputado, res_iso, features),
                  "iforest_anomalias")
    LOG.info("Isolation Forest: taxa de anomalias=%.3f (%d pontos)",
             res_iso.taxa, int(res_iso.flags.sum()))

    saida.to_csv(config.PROCESSED_DIR / "dados_rotulados.csv",
                 index=False, encoding="utf-8")
    LOG.info("Salvo data/processed/dados_rotulados.csv")


if __name__ == "__main__":
    main()
