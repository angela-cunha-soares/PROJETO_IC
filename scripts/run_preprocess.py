"""Pipeline de pré-processamento: carga → tabela de modelagem → imputação → padronização.

Saídas:
    * ``data/processed/dados_modelagem.csv``  — imputado (não padronizado), com data e features mantidas.
    * ``data/processed/dados_scaled.csv``     — versão padronizada (z-score) das features mantidas.
    * ``models/scaler.joblib``                — StandardScaler ajustado.
    * ``reports/tables/faltantes_resumo.csv`` — % faltantes + decisão por variável.
    * ``reports/tables/validacao_imputacao.csv`` — estatísticas antes/depois.

Uso:
    python scripts/run_preprocess.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
from features.build_features import tabela_modelagem  # noqa: E402
from models.reports import salvar_tabela  # noqa: E402
from preprocessing.missing import (  # noqa: E402
    impacto_kmeans, imputar, little_mcar_test, planejar_imputacao, validar_imputacao,
)
from preprocessing.scaling import padronizar, salvar_scaler  # noqa: E402
from projeto_pcj.load import load_semae  # noqa: E402
from projeto_pcj.schema import FEATURES  # noqa: E402

LOG = logging.getLogger("run_preprocess")
ID_COLS = ["data", "Ano", "Mes", "mes_num", "trimestre", "estacao"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config.ensure_dirs()

    LOG.info("Carregando base SEMAE…")
    df = load_semae()
    modelagem = tabela_modelagem(df, estatistica="Méd.")
    LOG.info("Tabela de modelagem: %d linhas × %d variáveis", len(modelagem), len(FEATURES))

    # Diagnóstico MCAR (informativo)
    mcar = little_mcar_test(modelagem, FEATURES)
    LOG.info("Teste de Little (MCAR): chi2=%s, df=%s, p=%s",
             mcar["chi2"], mcar["df"], mcar["p_value"])

    # Plano de imputação e execução
    plano = planejar_imputacao(modelagem, FEATURES)
    LOG.info("Excluir (>30%%): %s", plano.excluir)
    LOG.info("KNN (5-30%%): %s", plano.knn)
    LOG.info("Média: %s | Mediana: %s", plano.media, plano.mediana)

    imputado, plano = imputar(modelagem, FEATURES, plano=plano)
    mantidas = plano.manter

    validacao = validar_imputacao(modelagem, imputado, mantidas)

    # Validação adicional (projeto §3.1): impacto da imputação na silhueta do K-Means.
    impacto = impacto_kmeans(modelagem, imputado, mantidas)
    LOG.info("Impacto na silhueta do K-Means (complete-case vs imputado):\n%s",
             impacto.to_string(index=False))

    # Persistência
    salvar_tabela(plano.resumo, "faltantes_resumo")
    salvar_tabela(validacao, "validacao_imputacao")
    salvar_tabela(impacto, "validacao_imputacao_kmeans")

    imputado.to_csv(config.PROCESSED_CSV, index=False, encoding="utf-8")
    LOG.info("Salvo %s", config.PROCESSED_CSV)

    scaled, scaler = padronizar(imputado, mantidas)
    scaled.to_csv(config.PROCESSED_DIR / "dados_scaled.csv", index=False, encoding="utf-8")
    salvar_scaler(scaler, config.MODELS_DIR / "scaler.joblib")
    LOG.info("Padronização concluída — %d features mantidas", len(mantidas))


if __name__ == "__main__":
    main()
