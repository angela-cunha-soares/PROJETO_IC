"""Schema canônico das 23 variáveis físico-químicas/bacteriológicas extraídas do PDF SEMAE.

Fonte: data/interim/dados_organizados.csv (saída de scripts/extrair_dados.py).
Detalhamento humano: docs/dicionario_de_dados.md.
"""
from __future__ import annotations

# --- Colunas de identificação --------------------------------------------------

ID_COLUMNS: list[str] = ["Ano", "Mes", "Calc"]

#: Valores válidos para a coluna ``Mes``.
MESES: list[str] = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez",
    "Ano",
]

#: Valores válidos para a coluna ``Calc``.
ESTATISTICAS: list[str] = ["Mín.", "Méd.", "Máx."]


# --- Colunas de medição --------------------------------------------------------

#: Variáveis físico-químicas e bacteriológicas, na mesma ordem do CSV.
FEATURES: list[str] = [
    "COR(ppm Pt Co)", "TURB.(FTU)", "pH", "ALC.(ppm CaCO3)", "AC.(ppm CaCO3)",
    "O.C.(ppm O2)", "DBO(ppm O2)", "O.D.(ppm O2)", "Cl-(ppm Cl-)", "DUR.(ppm CaCO3)",
    "Fe(ppm Fe)", "Mn(mg/l)", "N(ppm N)", "P(ppm P)", "Cond.(uS/cm)", "Amonia(mg/l)",
    "Surfact.(mg/l)", "Cianobacteria(cel/ml)", "S.T.(mg/l)", "C.T.(NMP/100ml)",
    "C.F.(NMP/100ml)", "CLOROFILA(ug/l)", "F(ppm F)",
]

#: Agrupamentos temáticos das variáveis (úteis para EDA e seleção de features).
FEATURE_GROUPS: dict[str, list[str]] = {
    "fisica": ["COR(ppm Pt Co)", "TURB.(FTU)", "Cond.(uS/cm)", "S.T.(mg/l)"],
    "iao_basico": ["pH", "ALC.(ppm CaCO3)", "AC.(ppm CaCO3)", "DUR.(ppm CaCO3)"],
    "oxigenio": ["O.C.(ppm O2)", "DBO(ppm O2)", "O.D.(ppm O2)"],
    "nutrientes": ["N(ppm N)", "P(ppm P)", "Amonia(mg/l)"],
    "metais": ["Fe(ppm Fe)", "Mn(mg/l)"],
    "anions": ["Cl-(ppm Cl-)", "F(ppm F)"],
    "biologico": [
        "Cianobacteria(cel/ml)", "C.T.(NMP/100ml)",
        "C.F.(NMP/100ml)", "CLOROFILA(ug/l)",
    ],
    "outros": ["Surfact.(mg/l)"],
}


# --- Tipagem -------------------------------------------------------------------

#: Dtypes pandas por coluna. Variáveis numéricas ficam como ``float64`` (admitem NaN).
DTYPES: dict[str, str] = {
    "Ano": "int64",
    "Mes": "category",
    "Calc": "category",
    **{c: "float64" for c in FEATURES},
}


# --- Convenções do PDF SEMAE ---------------------------------------------------

#: Marcadores que o SEMAE usa para indicar ausência de medição no PDF.
MISSING_MARKERS: list[str] = ["--", "---", "-", ""]


# --- Limites regulatórios (CONAMA 357/2005, Classe 2) --------------------------
# Convenção: tuple (mín, máx). Use ``None`` quando o limite for unilateral ou inexistente.
# Detalhes e ressalvas em docs/dicionario_de_dados.md.

CONAMA_357_CLASSE2: dict[str, tuple[float | None, float | None]] = {
    "COR(ppm Pt Co)": (None, 75.0),
    "TURB.(FTU)": (None, 100.0),
    "pH": (6.0, 9.0),
    "DBO(ppm O2)": (None, 5.0),
    "O.D.(ppm O2)": (5.0, None),
    "Cl-(ppm Cl-)": (None, 250.0),
    "Fe(ppm Fe)": (None, 0.3),
    "Mn(mg/l)": (None, 0.1),
    # N(ppm N) agrega múltiplas formas; ver dicionario_de_dados.md
    "P(ppm P)": (None, 0.1),
    "Amonia(mg/l)": (None, 3.7),  # pH <= 7.5
    "Surfact.(mg/l)": (None, 0.5),
    "Cianobacteria(cel/ml)": (None, 50_000.0),
    "C.F.(NMP/100ml)": (None, 1_000.0),
    "CLOROFILA(ug/l)": (None, 30.0),
    "F(ppm F)": (None, 1.4),
}


# --- "Schema" pronto para uso (objeto único, compat. com README) ---------------

#: Estrutura agregada para consumo externo (ex.: notebooks).
SCHEMA: dict[str, object] = {
    "id_columns": ID_COLUMNS,
    "features": FEATURES,
    "feature_groups": FEATURE_GROUPS,
    "dtypes": DTYPES,
    "missing_markers": MISSING_MARKERS,
    "conama_357_classe2": CONAMA_357_CLASSE2,
}

__all__ = [
    "ID_COLUMNS", "MESES", "ESTATISTICAS",
    "FEATURES", "FEATURE_GROUPS",
    "DTYPES", "MISSING_MARKERS",
    "CONAMA_357_CLASSE2", "SCHEMA",
]
