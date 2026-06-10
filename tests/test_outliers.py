"""Testes da detecção de outliers (IQR e ensemble)."""
import numpy as np
import pandas as pd

from preprocessing.outliers import ensemble_outliers, iqr_mask, jaccard


def test_iqr_detecta_extremo():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000.0])
    mask = iqr_mask(s)
    assert bool(mask.iloc[-1]) is True
    assert mask.iloc[:-1].sum() == 0


def test_iqr_ignora_nan():
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 1000.0])
    mask = iqr_mask(s)
    assert mask.isna().sum() == 0  # NaN vira False
    assert len(mask) == len(s)


def test_ensemble_shape_e_votos():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    X[:5] += 12  # 5 anomalias claras
    res = ensemble_outliers(X, contamination=0.07)
    assert res.votos.shape == (100,)
    assert res.outliers.dtype == bool
    assert res.por_metodo.shape == (100, 3)
    assert res.outliers.sum() >= 1  # detecta algo


def test_jaccard_limites():
    a = np.array([True, False, True, False])
    assert jaccard(a, a) == 1.0
    b = np.array([False, True, False, True])
    assert jaccard(a, b) == 0.0
