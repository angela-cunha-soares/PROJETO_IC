"""Configuração central do projeto PCJ-ML.

Fonte única de caminhos, sementes, limiares e hiperparâmetros. Todo módulo do
pipeline (``preprocessing``, ``features``, ``models``, ``visualization`` e os
scripts em ``scripts/``) deve importar daqui em vez de redefinir constantes.

As decisões por trás destes valores estão documentadas em
``docs/decisoes_de_projeto.md`` (ADRs) e ``docs/metodologia.md``.
"""
from __future__ import annotations

from pathlib import Path

# --- Caminhos ------------------------------------------------------------------
# config.py vive em src/; a raiz do projeto é o avô deste arquivo.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"

RAW_PDF: Path = RAW_DIR / "data.pdf"
INTERIM_CSV: Path = INTERIM_DIR / "dados_organizados.csv"
PROCESSED_CSV: Path = PROCESSED_DIR / "dados_modelagem.csv"

# --- Reprodutibilidade ---------------------------------------------------------
RANDOM_STATE: int = 42

# --- Imputação (ADR-004) -------------------------------------------------------
#: Acima deste percentual de faltantes a variável é descartada.
MISSING_DROP_ABOVE: float = 0.30
#: Abaixo deste percentual usa-se média/mediana simples.
MISSING_SIMPLE_BELOW: float = 0.05
#: Entre SIMPLE_BELOW e DROP_ABOVE usa-se KNN.
#: Acima deste percentual de faltantes, padronizar antes do KNN.
KNN_SCALE_ABOVE: float = 0.15
#: |skew| acima deste valor → distribuição assimétrica → mediana em vez de média.
SKEW_THRESHOLD: float = 1.0
KNN_N_NEIGHBORS: int = 5
KNN_WEIGHTS: str = "distance"

# --- Detecção de outliers / anomalias (ADR-006, ADR-007) -----------------------
CONTAMINATION: float = 0.07
IQR_K: float = 1.5
LOF_N_NEIGHBORS: int = 20

# --- K-Means (ADR-006) ---------------------------------------------------------
KMEANS_K_RANGE: range = range(2, 11)
SILHOUETTE_GOOD: float = 0.5

# --- Random Forest (ADR-006) ---------------------------------------------------
RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 10
RF_TEST_SIZE: float = 0.20
RF_CV_FOLDS: int = 5

# --- Isolation Forest (ADR-006) ------------------------------------------------
IFOREST_N_ESTIMATORS: int = 200
IFOREST_CONTAMINATION: float = 0.07

# --- Estilo de figuras ---------------------------------------------------------
FIG_DPI: int = 120
FIG_FORMAT: str = "png"


def ensure_dirs() -> None:
    """Cria as pastas de saída do pipeline se ainda não existirem."""
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "INTERIM_DIR", "PROCESSED_DIR",
    "EXTERNAL_DIR", "MODELS_DIR", "REPORTS_DIR", "FIGURES_DIR", "TABLES_DIR",
    "RAW_PDF", "INTERIM_CSV", "PROCESSED_CSV", "RANDOM_STATE",
    "MISSING_DROP_ABOVE", "MISSING_SIMPLE_BELOW", "KNN_SCALE_ABOVE",
    "SKEW_THRESHOLD", "KNN_N_NEIGHBORS", "KNN_WEIGHTS",
    "CONTAMINATION", "IQR_K", "LOF_N_NEIGHBORS",
    "KMEANS_K_RANGE", "SILHOUETTE_GOOD",
    "RF_N_ESTIMATORS", "RF_MAX_DEPTH", "RF_TEST_SIZE", "RF_CV_FOLDS",
    "IFOREST_N_ESTIMATORS", "IFOREST_CONTAMINATION",
    "FIG_DPI", "FIG_FORMAT", "ensure_dirs",
]
