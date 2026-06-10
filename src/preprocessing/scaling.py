"""Padronização (StandardScaler) com persistência via joblib.

Algoritmos sensíveis a escala (K-Means, KNN, LOF) exigem média 0 e desvio 1.
O scaler é salvo para garantir que treino, validação e novos dados usem
exatamente a mesma transformação (reprodutibilidade — ADR-006).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def padronizar(
    df: pd.DataFrame, features: list[str], *, scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """Padroniza ``features`` (média 0, desvio 1) e devolve ``(df, scaler)``.

    Se ``scaler`` for fornecido (já ajustado), apenas aplica ``transform`` —
    útil para aplicar a transformação do treino a um conjunto de teste.
    """
    out = df.copy()
    X = out[features].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    out[features] = X
    return out, scaler


def salvar_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """Serializa o scaler para reuso (``joblib``)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def carregar_scaler(path: str | Path) -> StandardScaler:
    """Recarrega um scaler salvo por :func:`salvar_scaler`."""
    return joblib.load(path)


def matriz_scaled(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Atalho: devolve apenas a matriz numpy padronizada (sem persistência)."""
    return StandardScaler().fit_transform(df[features].to_numpy(dtype=float))


__all__ = ["padronizar", "salvar_scaler", "carregar_scaler", "matriz_scaled"]
