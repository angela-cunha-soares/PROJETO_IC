"""Testes da carga e do parser numérico brasileiro."""
import numpy as np

from projeto_pcj.load import _parse_br_number, faltantes_por_variavel
from projeto_pcj.schema import FEATURES, MESES


def test_parse_decimal_virgula():
    assert _parse_br_number("3,23") == 3.23


def test_parse_milhar_ponto():
    assert _parse_br_number("23.000") == 23000.0
    assert _parse_br_number("2.400.000") == 2_400_000.0


def test_parse_milhar_e_decimal():
    assert _parse_br_number("30.000,00") == 30000.0


def test_parse_decimal_nativo_sem_virgula():
    # "7.2" deve ser 7.2 (decimal), não 72 (milhar)
    assert _parse_br_number("7.2") == 7.2


def test_parse_marcadores_ausencia():
    for marcador in ("--", "---", "-", "", None):
        assert np.isnan(_parse_br_number(marcador))


def test_load_estrutura(df_semae):
    # 16 anos × 13 grupos (12 meses + Ano) × 3 estatísticas = 624
    assert len(df_semae) == 624
    assert {"Ano", "Mes", "Calc"}.issubset(df_semae.columns)
    for col in FEATURES:
        assert col in df_semae.columns


def test_load_dtypes(df_semae):
    assert df_semae["Ano"].dtype == "int64"
    assert df_semae["pH"].dtype == "float64"
    assert set(df_semae["Mes"].unique()).issubset(set(MESES))


def test_faltantes_ordenado(df_semae):
    falt = faltantes_por_variavel(df_semae)
    assert (falt.values[:-1] >= falt.values[1:]).all()  # desc
    assert (falt >= 0).all() and (falt <= 1).all()
