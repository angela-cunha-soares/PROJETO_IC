# Metodologia detalhada — Qualidade da água PCJ (ML)

Documento que detalha **o que fazer, com qual técnica e qual algoritmo**, em cada
etapa do pipeline, com as fórmulas declaradas no projeto FAPESP. Cada etapa aponta
o módulo (`src/`), o script (`scripts/`) e o notebook (`notebooks/`) correspondentes.

> Estrutura de dados: o PDF do SEMAE traz, por ano (2009–2024), para cada mês,
> três linhas — `Mín.`, `Méd.`, `Máx.`. A **tabela de modelagem** usa a estatística
> `Méd.` mensal (`features.build_features.tabela_modelagem`), resultando em
> **192 amostras** (16 anos × 12 meses) × 23 variáveis.

---

## Visão geral do pipeline

```
extrair_dados.py → load_semae → tabela_modelagem
        → [1] análise de faltantes (MCAR) → [2] imputação → [3] padronização
        → [4] análise exploratória → [5] outliers (IQR vs ensemble)
        → [6] K-Means → [7] Random Forest → [8] Isolation Forest
        → [9] validação consolidada → [10] relatório
```

Comando único de ponta a ponta:

```bash
python scripts/run_preprocess.py   # carga → imputação → padronização
python scripts/run_train.py        # K-Means, Random Forest, Isolation Forest
python scripts/run_evaluate.py     # métricas + figuras + relatório
pytest                             # testes
```

---

## Etapa 0 — Extração do PDF → CSV

- **Técnica:** parsing de tabelas com `pdfplumber`, mapeamento de cabeçalhos por
  normalização (`re.sub(r'[^a-z0-9]', '', s.lower())`) + dicionário (`HEADER_MAP`).
- **Por quê:** o layout muda entre anos (2011 reordena e renomeia colunas; 2018–2020
  trocam `C.F.`→`E COLI`; 2021–2024 inserem `Amônia`). Ver ADR-001/002.
- **Código:** `scripts/extrair_dados.py` → `data/interim/dados_organizados.csv`.
- **Verificação:** conferência célula a célula contra o PDF (2009, 2011, 2021) — OK.

## Etapa 1 — Carga e inspeção

- **Técnica:** parser numérico brasileiro tolerante (vírgula decimal, ponto de milhar,
  marcadores `--/---/-` → `NaN`), tipagem canônica (`Ano`:int, `Mes`/`Calc`:category).
- **Código:** `src/projeto_pcj/load.py` (`load_semae`), `schema.py`.
- **Notebook:** `01_carga_e_inspecao.ipynb`.

## Etapa 2 — Faltantes e suposição MCAR

- **Técnica:** percentual de faltantes por variável + **teste de Little** para a
  hipótese MCAR (`preprocessing.missing.little_mcar_test`).
- **Resultado obtido:** χ² ≈ 327,6 (df=269), **p ≈ 0,008 < 0,05 → rejeita-se MCAR**.
  Os faltantes **não** são completamente aleatórios (coerente com a cautela do
  `resumo_em_camadas.md` §5: sensores falham mais em eventos extremos). A imputação
  segue as regras declaradas, mas o relatório deve registrar esta limitação.
- **Notebook:** `02_pre_processamento.ipynb`.

## Etapa 3 — Imputação (ADR-004)

Árvore de decisão por variável (`preprocessing.missing.planejar_imputacao`):

| % faltantes | Ação | Algoritmo |
|---|---|---|
| > 30% | excluir | — |
| 5–30% | imputar | `KNNImputer(n_neighbors=5, weights="distance")`, padronizado se >15% |
| < 5%, simétrica (\|skew\|≤1) | imputar | média |
| < 5%, assimétrica (\|skew\|>1) | imputar | mediana |

- **Excluídas (>30%):** S.T., Surfact., DBO, P, N, Amônia, C.T. → restam **16 variáveis**.
  ⚠️ Conflito conhecido (ADR-004): N, P e Amônia são justamente as mais relevantes
  para irrigação. Mitigação: análise de sensibilidade / split temporal 2018–2024.
- **Validação:** comparação média/mediana/desvio antes×depois (`validar_imputacao`).
- **Notebook:** `03_imputacao.ipynb`.

## Etapa 4 — Padronização

- **Técnica:** `StandardScaler` (média 0, desvio 1) antes de algoritmos sensíveis a
  escala (K-Means, KNN, LOF). Persistido em `models/scaler.joblib`.
- **Código:** `src/preprocessing/scaling.py`.

## Etapa 5 — Análise exploratória

- **Técnica:** estatísticas descritivas, histogramas, boxplots (z-score), série
  temporal e **heatmap de correlação de Pearson** (+ Spearman para assimétricas).
- **Código:** `src/visualization/plots.py`. **Notebook:** `04_analise_exploratoria.ipynb`.

## Etapa 6 — Detecção de outliers (estudo comparativo)

> **Nota — não confundir duas lentes diferentes (fácil de errar no relatório):**
> 1. **Outlier estatístico** = ponto atípico *nos dados* (IQR, MAD, IForest, LOF, KNN).
>    Não usa CONAMA. Aqui ML multivariado é **mais adequado para apontar observações
>    anômalas** (melhor calibração e precisão); IQR/MAD (univariados) servem para
>    **triagem por variável** e por isso marcam muito mais (~25%).
> 2. **Violação CONAMA** = checagem **determinística** `valor > limite` (`if`), não
>    "detecção". Nenhum método de ML/tradicional a identifica — é uma regra legal.
>
> As duas lentes são **independentes e complementares**: outliers estatísticos e
> não conformidades tendem a coincidir na periferia dos dados (ver
> `outliers_por_metodo_normalizado.png`, 7º painel). O **outlier confirmado** é a
> interseção delas + evento conhecido — não um método "achando" violações.

- **Baseline (univariado):** regra do IQR de Tukey — outlier se
  `x < Q1 − k·IQR` ou `x > Q3 + k·IQR` (k=1,5).
- **Ensemble (multivariado):** votação por maioria (≥2/3) de **Isolation Forest +
  Local Outlier Factor + KNN (PyOD)**, com `contamination=0,07`.
- **Comparação:** índice de Jaccard entre conjuntos de flags + distribuição temporal
  (alinhamento com a crise do Cantareira 2014–2016 como validação externa empírica).
- **Código:** `src/preprocessing/outliers.py`, `scripts/comparar_metodos_outliers.py`,
  `scripts/melhor_metodo_por_variavel.py`. **Notebook:** `05_outliers_iqr_vs_ensemble.ipynb`.

### Critério de "outlier confirmado" (sem ground truth)

A detecção é não supervisionada — não há rótulo verdadeiro. Para decidir o que é
**realmente** anomalia (e não variação natural), combina-se:

1. **Consenso entre métodos** — marcado por ≥ 4 dos 6 (IQR, MAD, IForest, LOF, KNN, Ensemble);
2. **Âncora de domínio** — viola ≥ 1 limite da CONAMA 357/2005 Classe 2;
3. **Validação externa** — coincidência com eventos conhecidos (crise do Cantareira 2014–16).

- **Tabelas:** `scripts/tabela_outliers_por_metodo.py` gera, em `data/interim/`,
  `outliers_valores_por_metodo.csv` (valores + flag por método), `outliers_marcados_long.csv`
  (só marcados, com a variável mais extrema) e `outliers_confirmados.csv`
  (os que passam no critério acima — também em `reports/tables/`).
- **Figuras:** `scripts/plot_outliers_por_metodo.py` (painel PCA-2D por método) e
  `scripts/plot_outliers_confirmados.py` (`outliers_confirmados_heatmap.png` = mês ×
  parâmetro CONAMA violado; `outliers_confirmados_timeline.png` = linha do tempo).
- **Resultado:** 12 de 180 observações confirmadas — 8 delas em 2014/2016 (crise do
  Cantareira), validando a abordagem. Limites CONAMA: `src/projeto_pcj/schema.py`
  (`CONAMA_357_CLASSE2`), verificados contra o Art. 15 da Resolução 357/2005 e
  descritos em `docs/dicionario_de_dados.md`.
- **Ressalva (Ferro):** o Fe aparece como violador em quase todos os meses confirmados,
  mas a norma limita o Fe **dissolvido** (0,3 mg/L) e o SEMAE mede Fe **total** — logo
  as violações de Fe estão **superestimadas**. Já o Mn (0,1) é limite de Mn **total** e
  coincide com a medição. `violacoes_conama(..., excluir_indicativos=True)` desconta o Fe
  (lista `PARAMS_INDICATIVOS`); a coluna `n_violacoes_sem_indicativos` mostra que, mesmo
  sem o Fe, os 12 meses confirmados ainda violam 4–5 outros parâmetros — a confirmação
  **não depende** do parâmetro enviesado. Figuras: `outliers_confirmados_heatmap.png`
  (Fe hachurado como indicativo) e `outliers_confirmados_com_sem_fe.png` (com vs. sem Fe).

## Etapa 7 — K-Means (clustering)

- **Seleção de k:** método do cotovelo (WCSS) + coeficiente de silhueta, `k ∈ [2,10]`.
- **Silhueta** (Eq. 1, Rousseeuw 1987):

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}$$

onde `a(i)` é a distância média intra-cluster e `b(i)` a menor distância média a outro
cluster. Silhueta média > 0,5 indica clusters efetivos.

- **WCSS** (Eq. 2, Jain & Dubes 1988):

$$\mathrm{WCSS} = \sum_{i=1}^{k} \sum_{x \in C_i} \lVert x - \mu_i \rVert^2$$

- **Resultado obtido:** k=2 pela silhueta (cotovelo sugere 3). Interpretação dos grupos
  via `kmeans.perfil_clusters` (média de cada variável por cluster).
- **Código:** `src/models/kmeans.py`. **Notebook:** `06_kmeans_clustering.ipynb`.

## Etapa 8 — Random Forest (classificação)

- **Rotulagem (ADR-005):** limites da **CONAMA 357/2005 Classe 2** (`schema.CONAMA_357_CLASSE2`).
  - ⚠️ **Achado metodológico:** o rótulo estrito "viola qualquer parâmetro" é
    **degenerado** em água bruta (as 192 amostras dão `inadequada`), pois a norma
    classifica o corpo d'água, não a água tratada.
  - **Alvo usado:** *severidade* — nº de parâmetros violados binarizado pela mediana
    (`rotular_severidade`) → classes balanceadas (≈126 baixa / 66 alta).
- **Anti-leakage (ADR-005):** variáveis usadas na regra de rótulo são removidas das
  features (`features_sem_vazamento`) → o RF usa apenas as não normatizadas
  (ALC, AC, O.C., DUR, Cond).
- **Treino:** split 80/20 estratificado, `n_estimators=100`, `max_depth=10`,
  `class_weight="balanced"`, validação cruzada k=5.
- **Métricas** (Eq. 3 e 4):

$$\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN}, \qquad
F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$$

com `Precisão = VP/(VP+FP)` e `Recall = VP/(VP+FN)`.

- **Código:** `src/models/random_forest.py`, `src/models/metrics.py`.
  **Notebook:** `07_random_forest.ipynb`.

## Etapa 9 — Isolation Forest (anomalias)

- **Técnica:** isolamento por árvores aleatórias (Liu et al. 2008); caminhos curtos =
  anomalias. `contamination=0,07`, `n_estimators=200`, sobre dados padronizados.
- **Resultado obtido:** ~7,3% de anomalias (14/192). Recomenda-se calibrar
  `contamination` contra eventos conhecidos e considerar janelas sazonais (ADR-007).
- **Código:** `src/models/isolation_forest.py`. **Notebook:** `08_isolation_forest.ipynb`.

## Etapa 10 — Validação consolidada e relatório

- **Quantitativa:** silhueta + WCSS (K-Means); acurácia, F1, matriz de confusão e
  importância de variáveis (RF); taxa de anomalias (IForest).
- **Visual:** curva cotovelo/silhueta, scatter de clusters, matriz de confusão,
  scatter de anomalias, heatmap de correlação — todos em `reports/figures/`.
- **Saídas:** `reports/tables/*.csv`, `reports/relatorio_resultados.md`.
- **Notebooks:** `09_validacao_consolidada.ipynb`, `10_relatorio_final.ipynb`.

---

## Reprodutibilidade

- `random_state=42` em todos os modelos (ADR-006). Para robustez, repetir com várias
  sementes e reportar média ± desvio.
- Hashes SHA-256 do PDF e do CSV registrados em `docs/decisoes_de_projeto.md` (ADR-008).
- Ambiente fixado em `requirements.txt` / `pyproject.toml`; testes em `tests/` (pytest).
