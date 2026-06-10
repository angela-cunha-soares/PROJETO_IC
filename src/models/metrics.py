"""Métricas de validação dos modelos (seção 3.5 do projeto).

Reúne as equações declaradas no projeto FAPESP:

* Silhueta (Eq. 1, Rousseeuw 1987) e WCSS (Eq. 2, Jain & Dubes 1988) — K-Means;
* Acurácia (Eq. 3) e F1-Score (Eq. 4, Powers 2020) — Random Forest;
* taxa de anomalias — Isolation Forest.

São finos invólucros sobre o scikit-learn, centralizados para uso consistente
em notebooks, scripts e testes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)


# --- K-Means -------------------------------------------------------------------

def silhueta(X: np.ndarray, labels: np.ndarray) -> float:
    """Coeficiente de silhueta médio (Eq. 1). > 0,5 indica clusters efetivos."""
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Soma dos quadrados intra-cluster (Eq. 2)."""
    total = 0.0
    for k, c in enumerate(centroids):
        pts = X[labels == k]
        if len(pts):
            total += float(((pts - c) ** 2).sum())
    return total


# --- Random Forest -------------------------------------------------------------

def metricas_classificacao(y_true, y_pred) -> dict[str, float]:
    """Acurácia, precisão, recall e F1 (Eq. 3 e 4)."""
    return {
        "acuracia": round(float(accuracy_score(y_true, y_pred)), 4),
        "precisao": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def matriz_confusao_df(y_true, y_pred, labels=(0, 1)) -> pd.DataFrame:
    """Matriz de confusão como ``DataFrame`` rotulado (linhas=real, colunas=predito)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    nomes = ["adequada", "inadequada"] if set(labels) == {0, 1} else [str(x) for x in labels]
    return pd.DataFrame(
        cm,
        index=[f"real_{n}" for n in nomes],
        columns=[f"pred_{n}" for n in nomes],
    )


# --- Isolation Forest ----------------------------------------------------------

def taxa_anomalias(flags: np.ndarray) -> float:
    """Fração de pontos sinalizados como anômalos."""
    flags = np.asarray(flags).astype(bool)
    return round(float(flags.mean()), 4) if flags.size else 0.0


__all__ = [
    "silhueta", "wcss", "metricas_classificacao",
    "matriz_confusao_df", "taxa_anomalias",
]
