"""Detecção de outliers: regra do IQR (baseline) vs. ensemble de ML.

Implementa o estudo comparativo previsto no objetivo específico do projeto:

* :func:`iqr_mask` — regra clássica de Tukey (univariada);
* :func:`ensemble_outliers` — votação por maioria de Isolation Forest, Local
  Outlier Factor e KNN (PyOD), capturando anomalias globais, locais e
  contextuais (Aggarwal, 2017).

Todos os detectores multivariados operam sobre dados **já padronizados**.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import config


def iqr_mask(serie: pd.Series, k: float = config.IQR_K) -> pd.Series:
    """Máscara booleana de outliers pela regra do IQR (Tukey).

    Outlier se ``x < Q1 - k·IQR`` ou ``x > Q3 + k·IQR``. ``NaN`` → ``False``.
    """
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return ((serie < low) | (serie > high)).fillna(False)


def iqr_outliers_tabela(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Contagem de outliers por variável segundo o IQR."""
    linhas = []
    for col in features:
        mask = iqr_mask(df[col])
        n = int(mask.sum())
        linhas.append(
            {
                "variavel": col,
                "n_outliers": n,
                "pct": round(n / df[col].notna().sum() * 100, 2)
                if df[col].notna().sum()
                else 0.0,
            }
        )
    return pd.DataFrame(linhas).sort_values("n_outliers", ascending=False)


@dataclass
class ResultadoEnsemble:
    """Saída de :func:`ensemble_outliers`."""

    votos: np.ndarray            # 0..3 votos por linha
    outliers: np.ndarray         # bool, votos >= min_votos
    por_metodo: pd.DataFrame     # flags individuais de cada detector


def ensemble_outliers(
    X: np.ndarray,
    *,
    contamination: float = config.CONTAMINATION,
    min_votos: int = 2,
    random_state: int = config.RANDOM_STATE,
) -> ResultadoEnsemble:
    """Detecta outliers por votação de IForest + LOF + KNN (PyOD).

    Parameters
    ----------
    X
        Matriz padronizada (sem NaN).
    contamination
        Fração esperada de anomalias (0,05–0,1 no projeto).
    min_votos
        Quantos detectores precisam concordar para marcar outlier (padrão 2/3).
    """
    n = X.shape[0]

    iforest = IsolationForest(contamination=contamination, random_state=random_state)
    flag_if = (iforest.fit_predict(X) == -1).astype(int)

    lof = LocalOutlierFactor(
        n_neighbors=min(config.LOF_N_NEIGHBORS, n - 1), contamination=contamination
    )
    flag_lof = (lof.fit_predict(X) == -1).astype(int)

    flag_knn = _knn_flags(X, contamination, random_state)

    por_metodo = pd.DataFrame(
        {"IsolationForest": flag_if, "LOF": flag_lof, "KNN": flag_knn}
    )
    votos = por_metodo.sum(axis=1).to_numpy()
    outliers = votos >= min_votos
    return ResultadoEnsemble(votos=votos, outliers=outliers, por_metodo=por_metodo)


def _knn_flags(X: np.ndarray, contamination: float, random_state: int) -> np.ndarray:
    """Flags do detector KNN. Usa PyOD se disponível; senão, fallback por distância."""
    try:
        from pyod.models.knn import KNN

        knn = KNN(contamination=contamination)
        knn.fit(X)
        return knn.labels_.astype(int)
    except Exception:  # pragma: no cover - fallback sem PyOD
        from sklearn.neighbors import NearestNeighbors

        k = min(config.KNN_N_NEIGHBORS, X.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k).fit(X)
        dist, _ = nn.kneighbors(X)
        score = dist[:, -1]
        limite = np.quantile(score, 1 - contamination)
        return (score > limite).astype(int)


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Índice de Jaccard entre dois conjuntos de outliers (interseção/união)."""
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    uniao = (a | b).sum()
    return float((a & b).sum() / uniao) if uniao else 1.0


def calibrar_contamination(
    X: np.ndarray,
    grid: tuple[float, ...] = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10),
    *,
    cv: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Seleciona ``contamination`` (0,05–0,1) por estabilidade em validação cruzada.

    Como não há rótulo verdadeiro, usa-se um proxy de robustez (Zhao et al., 2020):
    para cada valor de ``contamination``, reajusta-se o Isolation Forest em cada
    *fold* de treino (K-fold), prediz-se a base inteira e mede-se a **concordância
    média (Jaccard)** das flags entre os folds. Valor mais estável = mais confiável.

    Returns
    -------
    DataFrame
        Colunas ``contamination``, ``n_flags``, ``taxa`` e ``estabilidade_jaccard``,
        ordenadas por estabilidade desc. A primeira linha é a recomendação.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    linhas = []
    for c in grid:
        full = (
            IsolationForest(contamination=c, n_estimators=200,
                            random_state=random_state).fit_predict(X) == -1
        )
        fold_flags = []
        for tr, _ in kf.split(X):
            modelo = IsolationForest(
                contamination=c, n_estimators=200, random_state=random_state
            ).fit(X[tr])
            fold_flags.append(modelo.predict(X) == -1)
        js = [
            jaccard(fold_flags[i], fold_flags[j])
            for i in range(len(fold_flags))
            for j in range(i + 1, len(fold_flags))
        ]
        linhas.append({
            "contamination": c,
            "n_flags": int(full.sum()),
            "taxa": round(float(full.mean()), 4),
            "estabilidade_jaccard": round(float(np.mean(js)), 4) if js else float("nan"),
        })
    return (
        pd.DataFrame(linhas)
        .sort_values("estabilidade_jaccard", ascending=False)
        .reset_index(drop=True)
    )


__all__ = [
    "iqr_mask", "iqr_outliers_tabela", "ResultadoEnsemble",
    "ensemble_outliers", "jaccard", "calibrar_contamination",
]
