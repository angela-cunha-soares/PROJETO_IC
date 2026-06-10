"""Exporta a tabela com os VALORES de cada ponto marcado como outlier, por método.

Para as 180 observações da camada multivariada (10 variáveis com ≤5% de NaN,
complete-case), gera duas saídas em ``data/interim/``:

* ``outliers_valores_por_metodo.csv`` (largo): uma linha por mês, com os valores
  ORIGINAIS das variáveis + uma coluna booleana por método (IQR, MAD, IForest,
  LOF, KNN, Ensemble) + ``n_metodos`` (quantos marcaram) + ``n_violacoes_conama``.
* ``outliers_marcados_long.csv`` (longo): só os pontos marcados por ALGUM método,
  uma linha por (mês × método), com o valor que mais destoa — fácil de filtrar.

Assim dá para inspecionar visualmente se cada ponto vermelho é plausível e se
extrapola limites legais (âncora de domínio para decidir o que é "real").

Uso:
    python scripts/tabela_outliers_por_metodo.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

try:
    from pyod.models.knn import KNN as PyodKNN
except ImportError:  # pragma: no cover
    PyodKNN = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.build_features import violacoes_conama  # noqa: E402
from features.build_features import tabela_modelagem  # noqa: E402
from projeto_pcj.load import load_semae  # noqa: E402
from projeto_pcj.schema import FEATURES  # noqa: E402

LOG = logging.getLogger("tabela_outliers")
OUT_WIDE = PROJECT_ROOT / "data" / "interim" / "outliers_valores_por_metodo.csv"
OUT_LONG = PROJECT_ROOT / "data" / "interim" / "outliers_marcados_long.csv"
OUT_CONFIRMADOS = PROJECT_ROOT / "data" / "interim" / "outliers_confirmados.csv"
OUT_CONFIRMADOS_REL = PROJECT_ROOT / "reports" / "tables" / "outliers_confirmados.csv"

#: Critério de "outlier confirmado": consenso entre métodos + âncora de domínio.
CONSENSO_MIN = 4          # marcado por >= 4 dos 6 métodos
EXIGE_VIOLACAO_CONAMA = True

RNG = 42
CONTAMINATION = 0.07
MAX_MISSING_MULTI = 0.05
MAD_THRESHOLD = 3.5


def _flags_univ(M: np.ndarray, modo: str) -> np.ndarray:
    n, p = M.shape
    flag = np.zeros(n, dtype=bool)
    for j in range(p):
        col = M[:, j]
        if modo == "iqr":
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            flag |= (col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)
        else:
            med = np.median(col)
            mad = np.median(np.abs(col - med)) or 1e-9
            flag |= np.abs(0.6745 * (col - med) / mad) > MAD_THRESHOLD
    return flag


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    m = tabela_modelagem(load_semae())
    feats = [c for c in FEATURES if m[c].isna().mean() <= MAX_MISSING_MULTI]
    cc = m.dropna(subset=feats).reset_index(drop=True)
    M = StandardScaler().fit_transform(cc[feats].to_numpy(float))
    LOG.info("Camada multivariada: %d obs × %d variáveis", *M.shape)

    flags = {
        "IQR": _flags_univ(M, "iqr"),
        "MAD": _flags_univ(M, "mad"),
        "IForest": IsolationForest(contamination=CONTAMINATION, n_estimators=200,
                                   random_state=RNG).fit_predict(M) == -1,
        "LOF": LocalOutlierFactor(n_neighbors=20,
                                  contamination=CONTAMINATION).fit_predict(M) == -1,
    }
    if PyodKNN is not None:
        knn = PyodKNN(contamination=CONTAMINATION)
        knn.fit(M)
        flags["KNN"] = knn.labels_.astype(bool)
    else:  # pragma: no cover
        flags["KNN"] = np.zeros(len(M), dtype=bool)
    votos = sum(flags[k].astype(int) for k in ("IForest", "LOF", "KNN"))
    flags["Ensemble"] = votos >= 2

    # Tabela larga: identificação + valores ORIGINAIS + flags por método
    wide = cc[["data", "Ano", "Mes"] + feats].copy()
    for nome, mask in flags.items():
        wide[f"flag_{nome}"] = mask
    wide["n_metodos"] = sum(flags[k].astype(int) for k in flags)
    wide["n_violacoes_conama"] = violacoes_conama(cc).sum(axis=1).values
    # Contagem "de-enviesada": desconta parâmetros indicativos (Fe total vs dissolvido)
    wide["n_violacoes_sem_indicativos"] = (
        violacoes_conama(cc, excluir_indicativos=True).sum(axis=1).values
    )

    OUT_WIDE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_WIDE, index=False, encoding="utf-8")
    LOG.info("Salvo %s (%d linhas)", OUT_WIDE, len(wide))

    # Tabela longa: só os marcados, com a variável que mais destoa (|z| máximo)
    z = pd.DataFrame(M, columns=feats)
    registros = []
    for nome, mask in flags.items():
        for i in np.where(mask)[0]:
            zi = z.iloc[i].abs()
            var_top = zi.idxmax()
            registros.append({
                "Ano": int(cc.loc[i, "Ano"]),
                "Mes": cc.loc[i, "Mes"],
                "metodo": nome,
                "variavel_mais_extrema": var_top,
                "valor_original": round(float(cc.loc[i, var_top]), 3),
                "z_score": round(float(z.iloc[i][var_top]), 2),
                "n_metodos": int(wide.loc[i, "n_metodos"]),
                "n_violacoes_conama": int(wide.loc[i, "n_violacoes_conama"]),
            })
    longo = (pd.DataFrame(registros)
             .sort_values(["n_metodos", "Ano", "Mes"], ascending=[False, True, True]))
    longo.to_csv(OUT_LONG, index=False, encoding="utf-8")
    LOG.info("Salvo %s (%d marcações)", OUT_LONG, len(longo))

    # --- Outliers CONFIRMADOS (consenso + violação CONAMA) ----------------
    # Critério de domínio usa a contagem SEM indicativos (Fe descontado), para não
    # confirmar um outlier apenas por causa do viés de fração do Ferro.
    cond = wide["n_metodos"] >= CONSENSO_MIN
    if EXIGE_VIOLACAO_CONAMA:
        cond &= wide["n_violacoes_sem_indicativos"] > 0
    metodos = [f"flag_{k}" for k in flags]
    confirmados = (
        wide[cond]
        .assign(metodos_que_marcaram=lambda d: d[metodos].apply(
            lambda r: ", ".join(k.replace("flag_", "") for k in metodos if r[k]), axis=1))
        [["data", "Ano", "Mes", "n_metodos", "n_violacoes_conama",
          "n_violacoes_sem_indicativos", "metodos_que_marcaram"] + feats]
        .sort_values(["n_metodos", "n_violacoes_sem_indicativos", "data"],
                     ascending=[False, False, True])
        .reset_index(drop=True)
    )
    confirmados.to_csv(OUT_CONFIRMADOS, index=False, encoding="utf-8")
    OUT_CONFIRMADOS_REL.parent.mkdir(parents=True, exist_ok=True)
    confirmados.to_csv(OUT_CONFIRMADOS_REL, index=False, encoding="utf-8")
    LOG.info("Outliers confirmados (>=%d métodos%s): %d de %d obs → %s",
             CONSENSO_MIN, " + violação CONAMA" if EXIGE_VIOLACAO_CONAMA else "",
             len(confirmados), len(wide), OUT_CONFIRMADOS)
    LOG.info("\n%s", confirmados[["Ano", "Mes", "n_metodos", "n_violacoes_conama",
             "n_violacoes_sem_indicativos", "metodos_que_marcaram"]].to_string(index=False))


if __name__ == "__main__":
    main()
