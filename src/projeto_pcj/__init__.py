"""Projeto IC PCJ — Aplicação de ML à qualidade da água do rio Piracicaba.

Subpacote ``projeto_pcj`` reúne os módulos reutilizáveis usados pelos notebooks.
"""
from projeto_pcj.load import load_semae
from projeto_pcj.schema import FEATURES, SCHEMA

__all__ = ["load_semae", "FEATURES", "SCHEMA"]
