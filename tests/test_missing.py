"""Testes das regras de imputação (ADR-004)."""
import numpy as np

from preprocessing.missing import (
    imputar, planejar_imputacao, resumo_faltantes, validar_imputacao,
)


def test_decisao_por_faixa(df_sintetico):
    plano = planejar_imputacao(df_sintetico, ["a", "b", "c"])
    # c tem 70% de faltantes → excluir
    assert "c" in plano.excluir
    # a e b têm <5% faltantes (b: 0%, a: 10% → KNN)
    assert "a" in plano.knn  # 10% de faltantes cai na faixa KNN
    assert "b" in plano.media + plano.mediana


def test_imputar_remove_colunas_excluidas(df_sintetico):
    out, plano = imputar(df_sintetico, ["a", "b", "c"])
    assert "c" not in out.columns
    # Não deve sobrar NaN nas variáveis mantidas
    assert out[plano.manter].isna().sum().sum() == 0


def test_imputar_preserva_id():
    import pandas as pd

    # x com ~9% de faltantes (1/11) → imputado, não excluído
    df = pd.DataFrame({
        "id": list(range(11)),
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, None, 7.0, 8.0, 9.0, 10.0, 11.0],
    })
    out, _ = imputar(df, ["x"])
    assert "id" in out.columns
    assert out["x"].isna().sum() == 0


def test_resumo_faltantes_decisoes(df_sintetico):
    resumo = resumo_faltantes(df_sintetico, ["a", "b", "c"])
    assert set(resumo["decisao"]).issubset({"EXCLUIR", "Média", "Mediana", "KNN"})
    assert resumo["pct_nan"].max() <= 100


def test_validar_imputacao_sem_nan(df_sintetico):
    out, plano = imputar(df_sintetico, ["a", "b", "c"])
    val = validar_imputacao(df_sintetico, out, plano.manter)
    assert not val.empty
    assert "delta_media_%" in val.columns


def test_imputacao_na_base_real(df_modelagem):
    from projeto_pcj.schema import FEATURES

    out, plano = imputar(df_modelagem, FEATURES)
    assert out[plano.manter].isna().sum().sum() == 0
    assert len(plano.excluir) >= 1  # ao menos S.T./Surfact. saem
