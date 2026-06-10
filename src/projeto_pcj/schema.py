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
#
# Fonte: Resolução CONAMA nº 357/2005. Os limites próprios da Classe 2 estão no
# Art. 15 (águas doces); o Art. 15 ainda determina que "aplicam-se às águas doces
# de classe 2 as condições e padrões da classe 1, à exceção" dos itens listados —
# por isso os parâmetros não citados no Art. 15 herdam os limites da Classe 1
# (Art. 14, Tabela I).
# VERIFICADO valor a valor contra o texto oficial dos Art. 14 e Art. 15 (2026-06-10).
# Nota: 0,3 (Fe) e 0,1 (Mn) são Classe 1/2; a Classe 3 (Art. 16) é mais permissiva
# (Fe 5,0; Mn 0,5) e NÃO se aplica aqui.

CONAMA_357_CLASSE2: dict[str, tuple[float | None, float | None]] = {
    "COR(ppm Pt Co)": (None, 75.0),       # Art. 15, III — cor verdadeira 75 mg Pt/L
    "TURB.(FTU)": (None, 100.0),          # Art. 15, IV  — turbidez 100 UNT
    "pH": (6.0, 9.0),                     # Art. 14, I.m (Classe 1) — pH 6,0 a 9,0
    "DBO(ppm O2)": (None, 5.0),           # Art. 15, V   — DBO5,20 5 mg/L O2
    "O.D.(ppm O2)": (5.0, None),          # Art. 15, VI  — OD não inferior a 5 mg/L O2
    "Cl-(ppm Cl-)": (None, 250.0),        # Art. 14, Tab. I — cloreto total 250 mg/L
    "Fe(ppm Fe)": (None, 0.3),            # Art. 14, Tab. I — Ferro DISSOLVIDO 0,3 (SEMAE mede total: ressalva)
    "Mn(mg/l)": (None, 0.1),              # Art. 14, Tab. I — Manganês TOTAL 0,1 (coincide com SEMAE)
    # N(ppm N) agrega múltiplas formas (Art. 14: nitrato 10; nitrito 1); ver dicionario_de_dados.md
    "P(ppm P)": (None, 0.1),              # Art. 14, Tab. I — fósforo total lótico 0,1 mg/L (Art. 15 IX só trata lêntico/intermediário)
    "Amonia(mg/l)": (None, 3.7),          # Art. 14, Tab. I — N-amoniacal 3,7 mg/L N para pH <= 7,5
    "Surfact.(mg/l)": (None, 0.5),        # Art. 14, Tab. I — substâncias tensoativas (MBAS) 0,5 mg/L LAS
    "Cianobacteria(cel/ml)": (None, 50_000.0),  # Art. 15, VIII — 50.000 cel/mL
    "C.F.(NMP/100ml)": (None, 1_000.0),   # Art. 15, II — coliformes termotolerantes 1.000/100 mL
    "CLOROFILA(ug/l)": (None, 30.0),      # Art. 15, VII — clorofila a 30 µg/L
    "F(ppm F)": (None, 1.4),              # Art. 14, Tab. I — fluoreto total 1,4 mg/L F
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
