"""Análise e tratamento de valores ausentes (ADR-004).

Implementa as regras declaradas no projeto FAPESP:

* faltantes > 30%  → excluir variável;
* faltantes < 5%   → imputar média (simétrica) ou mediana (assimétrica);
* 5% – 30%         → imputar por KNN (``n_neighbors=5``, ``weights="distance"``),
  com padronização prévia quando faltantes > 15%.

A suposição declarada é MCAR (Lepot et al., 2017). A função :func:`little_mcar_test`
oferece um teste aproximado para criticar essa hipótese.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

import config


@dataclass
class PlanoImputacao:
    """Resultado de :func:`planejar_imputacao` — decisão por variável."""

    excluir: list[str] = field(default_factory=list)
    media: list[str] = field(default_factory=list)
    mediana: list[str] = field(default_factory=list)
    knn: list[str] = field(default_factory=list)
    resumo: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def manter(self) -> list[str]:
        """Variáveis que sobrevivem à regra dos 30% (todas menos as excluídas)."""
        return self.media + self.mediana + self.knn


def resumo_faltantes(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Tabela ``variavel × (pct_nan, skew, decisao)`` ordenada por faltantes desc."""
    pct = df[features].isna().mean() * 100
    skew = df[features].skew(numeric_only=True)
    linhas = []
    for col in features:
        decisao = _decidir(pct[col] / 100, skew.get(col, np.nan))
        linhas.append(
            {
                "variavel": col,
                "pct_nan": round(float(pct[col]), 2),
                "skew": round(float(skew.get(col, np.nan)), 2)
                if pd.notna(skew.get(col, np.nan))
                else np.nan,
                "decisao": decisao,
            }
        )
    return (
        pd.DataFrame(linhas)
        .sort_values("pct_nan", ascending=False)
        .reset_index(drop=True)
    )


def _decidir(frac_nan: float, skew: float) -> str:
    """Aplica a árvore de decisão da ADR-004 a uma variável."""
    if frac_nan > config.MISSING_DROP_ABOVE:
        return "EXCLUIR"
    if frac_nan < config.MISSING_SIMPLE_BELOW:
        if pd.notna(skew) and abs(skew) > config.SKEW_THRESHOLD:
            return "Mediana"
        return "Média"
    return "KNN"


def planejar_imputacao(df: pd.DataFrame, features: list[str]) -> PlanoImputacao:
    """Classifica cada variável em excluir / média / mediana / KNN."""
    resumo = resumo_faltantes(df, features)
    plano = PlanoImputacao(resumo=resumo)
    mapa = {"EXCLUIR": plano.excluir, "Média": plano.media,
            "Mediana": plano.mediana, "KNN": plano.knn}
    for _, row in resumo.iterrows():
        mapa[row["decisao"]].append(row["variavel"])
    return plano


def imputar(
    df: pd.DataFrame,
    features: list[str],
    *,
    plano: PlanoImputacao | None = None,
) -> tuple[pd.DataFrame, PlanoImputacao]:
    """Imputa faltantes conforme o plano e devolve ``(df_imputado, plano)``.

    As variáveis marcadas para exclusão são removidas do ``DataFrame`` retornado.
    Colunas de identificação (não presentes em ``features``) são preservadas.
    """
    if plano is None:
        plano = planejar_imputacao(df, features)

    out = df.copy()

    # Média / mediana (imputação univariada simples)
    for col in plano.media:
        out[col] = out[col].fillna(out[col].mean())
    for col in plano.mediana:
        out[col] = out[col].fillna(out[col].median())

    # KNN multivariado (usa todas as variáveis mantidas como vizinhança)
    if plano.knn:
        base_knn = plano.media + plano.mediana + plano.knn
        bloco = out[base_knn].to_numpy(dtype=float)

        max_frac = out[plano.knn].isna().mean().max()
        scaler = None
        if max_frac > config.KNN_SCALE_ABOVE:
            scaler = StandardScaler()
            bloco = scaler.fit_transform(bloco)

        imputer = KNNImputer(
            n_neighbors=config.KNN_N_NEIGHBORS, weights=config.KNN_WEIGHTS
        )
        bloco = imputer.fit_transform(bloco)

        if scaler is not None:
            bloco = scaler.inverse_transform(bloco)

        out[base_knn] = bloco

    # Remove variáveis descartadas pela regra dos 30%
    out = out.drop(columns=plano.excluir, errors="ignore")
    return out, plano


def validar_imputacao(
    antes: pd.DataFrame, depois: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Compara média/mediana/desvio antes e depois da imputação.

    Variações grandes sinalizam que a imputação distorceu a distribuição.
    """
    cols = [c for c in features if c in depois.columns]
    linhas = []
    for col in cols:
        a, d = antes[col], depois[col]
        linhas.append(
            {
                "variavel": col,
                "media_antes": a.mean(), "media_depois": d.mean(),
                "mediana_antes": a.median(), "mediana_depois": d.median(),
                "std_antes": a.std(), "std_depois": d.std(),
                "delta_media_%": _delta_pct(a.mean(), d.mean()),
                "delta_std_%": _delta_pct(a.std(), d.std()),
            }
        )
    return pd.DataFrame(linhas).round(3)


def _delta_pct(a: float, b: float) -> float:
    if a == 0 or pd.isna(a):
        return np.nan
    return round((b - a) / abs(a) * 100, 2)


def impacto_kmeans(
    df_original: pd.DataFrame,
    df_imputado: pd.DataFrame,
    features: list[str],
    *,
    k_range: range = range(2, 7),
) -> pd.DataFrame:
    """Mede o impacto da imputação na estrutura de clusters (validação 3.1 do projeto).

    Compara a melhor silhueta de um K-Means sobre (a) o subconjunto *complete-case*
    (linhas sem nenhum faltante) e (b) o conjunto imputado. Silhuetas próximas
    indicam que a imputação **preservou** a estrutura; queda grande sinaliza distorção.

    Returns
    -------
    DataFrame
        Linhas ``complete-case`` e ``imputado`` com ``n``, ``melhor_k`` e ``silhueta``.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in features if c in df_imputado.columns]

    def _melhor_sil(frame: pd.DataFrame) -> tuple[float, int]:
        sub = frame.dropna(subset=feats)
        if len(sub) <= max(k_range):
            return float("nan"), -1
        X = StandardScaler().fit_transform(sub[feats].to_numpy(float))
        melhor = (-1.0, -1)
        for k in k_range:
            labels = KMeans(
                n_clusters=k, n_init="auto", random_state=config.RANDOM_STATE
            ).fit_predict(X)
            s = float(silhouette_score(X, labels))
            if s > melhor[0]:
                melhor = (s, k)
        return melhor

    sil_cc, k_cc = _melhor_sil(df_original[feats])
    sil_imp, k_imp = _melhor_sil(df_imputado)
    return pd.DataFrame(
        [
            {"conjunto": "complete-case",
             "n": int(df_original[feats].dropna().shape[0]),
             "melhor_k": k_cc, "silhueta": round(sil_cc, 4)},
            {"conjunto": "imputado",
             "n": int(df_imputado.dropna(subset=feats).shape[0]),
             "melhor_k": k_imp, "silhueta": round(sil_imp, 4)},
        ]
    )


def little_mcar_test(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Teste de Little (aproximado) para a hipótese MCAR.

    Retorna estatística qui-quadrado, graus de liberdade e p-valor. p < 0,05
    rejeita MCAR (os faltantes não seriam completamente aleatórios).

    Implementação self-contained (não exige ``statsmodels``); segue a formulação
    de Little (1988) baseada em padrões de ausência e médias por padrão.
    """
    from scipy import stats

    data = df[features].copy()
    # Restringe a variáveis suficientemente medidas: covariância pareada com
    # variáveis de períodos disjuntos produz NaN e inviabiliza o teste.
    cols = [c for c in features if data[c].notna().mean() >= 0.5]
    if len(cols) < 2:
        return {"chi2": np.nan, "df": np.nan, "p_value": np.nan}
    data = data[cols]
    n_vars = len(cols)

    global_mean = data.mean()
    # cov pareada pode deixar NaN onde duas variáveis nunca coexistem → zera.
    global_cov = data.cov().fillna(0.0)
    # Regulariza a covariância para garantir inversibilidade.
    global_cov += np.eye(n_vars) * 1e-6
    try:
        inv_cov = np.linalg.pinv(global_cov.to_numpy())
    except np.linalg.LinAlgError:
        return {"chi2": np.nan, "df": np.nan, "p_value": np.nan}

    pattern = data.notna()
    pattern_id = pattern.apply(lambda r: tuple(r), axis=1)

    chi2 = 0.0
    dof = 0
    for _, idx in pattern_id.groupby(pattern_id).groups.items():
        sub = data.loc[idx]
        observed = list(sub.columns[sub.iloc[0].notna()])
        if not observed:
            continue
        pos = [cols.index(c) for c in observed]
        m_j = sub[observed].mean()
        diff = (m_j - global_mean[observed]).to_numpy()
        sub_inv = inv_cov[np.ix_(pos, pos)]
        chi2 += len(sub) * float(diff @ sub_inv @ diff)
        dof += len(observed)

    dof = max(dof - n_vars, 1)
    p = float(stats.chi2.sf(chi2, dof))
    return {"chi2": round(chi2, 3), "df": dof, "p_value": round(p, 5)}


__all__ = [
    "PlanoImputacao", "resumo_faltantes", "planejar_imputacao",
    "imputar", "validar_imputacao", "impacto_kmeans", "little_mcar_test",
]
