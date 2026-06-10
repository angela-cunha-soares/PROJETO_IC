"""Funções de visualização do pipeline (validação visual da seção 3.5).

Cada função salva uma figura em ``reports/figures/`` (se ``salvar_em`` for dado)
e devolve o ``Axes`` para uso interativo em notebooks. Estilo neutro, sem cores
fixas — segue o padrão do matplotlib/seaborn.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config


def _salvar(fig, salvar_em: str | Path | None) -> None:
    if salvar_em is not None:
        path = Path(salvar_em)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")


def histograma(df: pd.DataFrame, col: str, *, salvar_em=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribuição — {col}")
    _salvar(fig, salvar_em)
    return ax


def boxplots(df: pd.DataFrame, cols: list[str], *, salvar_em=None):
    """Boxplots padronizados (z-score) para comparar variáveis em escalas diferentes."""
    dados = df[cols].apply(lambda s: (s - s.mean()) / s.std())
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.8), 4))
    sns.boxplot(data=dados, ax=ax)
    ax.set_title("Boxplots (z-score) por variável")
    ax.tick_params(axis="x", rotation=90)
    _salvar(fig, salvar_em)
    return ax


def heatmap_correlacao(df: pd.DataFrame, cols: list[str], *, metodo="pearson", salvar_em=None):
    corr = df[cols].corr(method=metodo)
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 2, 0.6 * len(cols) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Correlação ({metodo})")
    _salvar(fig, salvar_em)
    return ax


def serie_temporal(df: pd.DataFrame, col: str, *, data_col="data", salvar_em=None):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(df[data_col], df[col], marker=".", lw=1)
    ax.set_title(f"Série temporal — {col}")
    ax.set_xlabel("data")
    ax.set_ylabel(col)
    _salvar(fig, salvar_em)
    return ax


def curva_cotovelo_silhueta(tabela: pd.DataFrame, *, salvar_em=None):
    """Plota WCSS (cotovelo) e silhueta lado a lado em função de k."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(tabela["k"], tabela["wcss"], marker="o")
    ax1.set(title="Método do cotovelo", xlabel="k", ylabel="WCSS")
    ax2.plot(tabela["k"], tabela["silhueta"], marker="o", color="tab:green")
    ax2.axhline(config.SILHOUETTE_GOOD, ls="--", color="gray", label="0,5")
    ax2.set(title="Coeficiente de silhueta", xlabel="k", ylabel="silhueta")
    ax2.legend()
    _salvar(fig, salvar_em)
    return (ax1, ax2)


def scatter_clusters(df, x, y, labels, *, salvar_em=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=df[x], y=df[y], hue=labels, palette="tab10", ax=ax, s=40)
    ax.set_title(f"Clusters — {x} × {y}")
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    _salvar(fig, salvar_em)
    return ax


def matriz_confusao(cm_df: pd.DataFrame, *, salvar_em=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Matriz de confusão")
    ax.set_xlabel("predito")
    ax.set_ylabel("real")
    _salvar(fig, salvar_em)
    return ax


def importancia_variaveis(importancias: pd.Series, *, top=15, salvar_em=None):
    imp = importancias.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(imp))))
    ax.barh(imp.index, imp.values)
    ax.set_title("Importância das variáveis (Random Forest)")
    _salvar(fig, salvar_em)
    return ax


def scatter_anomalias(df, x, y, flags, *, salvar_em=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x][~flags], df[y][~flags], s=25, label="normal", alpha=0.6)
    ax.scatter(df[x][flags], df[y][flags], s=55, color="red",
               label="anomalia", edgecolor="k")
    ax.set(title=f"Anomalias — {x} × {y}", xlabel=x, ylabel=y)
    ax.legend()
    _salvar(fig, salvar_em)
    return ax


__all__ = [
    "histograma", "boxplots", "heatmap_correlacao", "serie_temporal",
    "curva_cotovelo_silhueta", "scatter_clusters", "matriz_confusao",
    "importancia_variaveis", "scatter_anomalias",
]
