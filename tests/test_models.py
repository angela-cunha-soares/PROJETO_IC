"""Testes de modelos e métricas (K-Means, Random Forest, Isolation Forest)."""
import numpy as np
import pandas as pd

from models import isolation_forest as iso
from models import kmeans as km
from models.metrics import matriz_confusao_df, metricas_classificacao, silhueta
from features.build_features import (
    contar_violacoes, features_sem_vazamento, rotular_severidade, violacoes_conama,
)


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-5, size=(50, 2))
    b = rng.normal(loc=5, size=(50, 2))
    return np.vstack([a, b])


def test_kmeans_varredura_encontra_dois_grupos():
    X = _blobs()
    varredura = km.varrer_k(X, k_range=range(2, 6))
    assert varredura.k_silhueta == 2  # dois blobs bem separados
    assert (varredura.tabela["wcss"].values[:-1] >= varredura.tabela["wcss"].values[1:]).all()


def test_silhueta_alta_para_blobs():
    X = _blobs()
    modelo = km.ajustar(X, 2)
    assert silhueta(X, modelo.labels_) > 0.5


def test_metricas_classificacao_perfeita():
    y = [0, 0, 1, 1]
    m = metricas_classificacao(y, y)
    assert m["acuracia"] == 1.0 and m["f1"] == 1.0


def test_matriz_confusao_formato():
    cm = matriz_confusao_df([0, 1, 0, 1], [0, 1, 1, 1])
    assert cm.shape == (2, 2)
    assert cm.to_numpy().sum() == 4


def test_isolation_forest_detecta_anomalia():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    X[0] = [20, 20]
    res = iso.detectar(X, contamination=0.05)
    assert res.flags[0]  # ponto extremo é anomalia
    assert 0 < res.taxa < 0.2


def test_calibrar_contamination_retorna_grade():
    from preprocessing.outliers import calibrar_contamination

    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 3))
    X[:8] += 10
    tab = calibrar_contamination(X, grid=(0.05, 0.07, 0.10), cv=3)
    assert set(tab["contamination"]) == {0.05, 0.07, 0.10}
    assert tab["estabilidade_jaccard"].between(0, 1).all()
    assert (tab["taxa"] > 0).all()


def test_impacto_kmeans_imputacao(df_modelagem):
    from preprocessing.missing import imputar, impacto_kmeans

    out, plano = imputar(df_modelagem, list(df_modelagem.columns[6:]))
    tab = impacto_kmeans(df_modelagem, out, plano.manter)
    assert set(tab["conjunto"]) == {"complete-case", "imputado"}
    assert tab["silhueta"].between(-1, 1).all()


def test_violacoes_conama_e_severidade(df_modelagem):
    viol = violacoes_conama(df_modelagem)
    assert viol.shape[0] == len(df_modelagem)
    n = contar_violacoes(df_modelagem)
    assert (n >= 0).all()
    rotulo, usadas = rotular_severidade(df_modelagem)
    assert rotulo.nunique() == 2  # alvo balanceado, 2 classes
    # anti-leakage: features do RF não contêm as variáveis da regra
    feats = features_sem_vazamento(list(df_modelagem.columns), usadas)
    assert not (set(feats) & set(usadas))
