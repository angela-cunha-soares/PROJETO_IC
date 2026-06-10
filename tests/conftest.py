"""Configuração compartilhada dos testes: coloca ``src/`` no ``sys.path``."""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session")
def df_semae():
    """Base SEMAE carregada (pulada se o CSV intermediário não existir)."""
    from projeto_pcj.load import load_semae

    csv = PROJECT_ROOT / "data" / "interim" / "dados_organizados.csv"
    if not csv.is_file():
        pytest.skip("dados_organizados.csv ausente — rode scripts/extrair_dados.py")
    return load_semae()


@pytest.fixture(scope="session")
def df_modelagem(df_semae):
    """Tabela de modelagem (Méd. mensal) derivada da base."""
    from features.build_features import tabela_modelagem

    return tabela_modelagem(df_semae)


@pytest.fixture
def df_sintetico():
    """DataFrame pequeno e controlado para testes determinísticos de imputação/outliers."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 100.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "c": [1.0, 1.0, 1.0, None, None, None, None, None, None, None],  # 70% nan
        }
    )
