"""Geração e persistência de artefatos de relatório (tabelas e modelos).

Centraliza a escrita em ``reports/tables/`` e ``models/`` para que notebooks e
scripts produzam saídas em locais previsíveis e versionáveis.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

import config


def salvar_tabela(df: pd.DataFrame, nome: str, *, index: bool = False) -> Path:
    """Salva um ``DataFrame`` em ``reports/tables/<nome>.csv``."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABLES_DIR / (nome if nome.endswith(".csv") else f"{nome}.csv")
    df.to_csv(path, index=index, encoding="utf-8")
    return path


def salvar_modelo(modelo: object, nome: str) -> Path:
    """Serializa um modelo treinado em ``models/<nome>.joblib``."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / (nome if nome.endswith(".joblib") else f"{nome}.joblib")
    joblib.dump(modelo, path)
    return path


def carregar_modelo(nome: str) -> object:
    """Recarrega um modelo salvo por :func:`salvar_modelo`."""
    path = config.MODELS_DIR / (nome if nome.endswith(".joblib") else f"{nome}.joblib")
    return joblib.load(path)


def resumo_markdown(secoes: dict[str, str]) -> str:
    """Monta um resumo em Markdown a partir de ``{titulo: conteudo}``."""
    partes = ["# Relatório — Qualidade da água PCJ (ML)\n"]
    for titulo, conteudo in secoes.items():
        partes.append(f"## {titulo}\n\n{conteudo}\n")
    return "\n".join(partes)


__all__ = [
    "salvar_tabela", "salvar_modelo", "carregar_modelo", "resumo_markdown",
]
