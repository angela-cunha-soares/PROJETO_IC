# Aplicação de Machine Learning para Análise da Qualidade da Água na Bacia do PCJ

Projeto de Iniciação Científica (FAPESP) — ESALQ/USP
Parcerias: CENA-USP, C4AI · Dados: SEMAE Piracicaba (2009–2024)

---

## Objetivo

Organizar e analisar uma base histórica de qualidade de água do rio Piracicaba (bacia PCJ) usando Machine Learning, gerando: (i) base limpa e documentada, (ii) padrões de qualidade (clustering), (iii) classificação de adequação para irrigação e (iv) detecção de anomalias críticas.

---

## Sumário

1. Requisitos e instalação
2. Estrutura do repositório
3. Pipeline geral (Figura 2 do projeto)
4. Passo a passo da metodologia em Python
5. Como rodar
6. Reprodutibilidade
7. Referências

---

## 1. Requisitos e instalação

- Python 3.11+
- Visual Studio Code (com extensão Jupyter) — editor oficial do projeto
- Git e conta no GitHub

```bash
# Clonar o repositório
git clone https://github.com/angela-cunha-soares/PROJETO_IC.git
cd PROJETO_IC

# Criar e ativar ambiente virtual
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
scipy>=1.13
matplotlib>=3.8
matplotlib-venn>=1.1
seaborn>=0.13
pyod>=2.0
joblib>=1.4
jupyterlab>=4.2
ipykernel>=6.29
missingno>=0.5
pdfplumber>=0.11
python-docx>=1.1
```

---

## 2. Estrutura do repositório

Notebooks numerados em `notebooks/` reproduzem o pipeline; o código reutilizável vive em `src/`.

```
PROJETO_IC/
├── data/           # raw, interim, processed, external
├── notebooks/      # 00..10 — passo a passo executável
├── src/            # módulos Python (projeto_pcj, preprocessing, features, models, visualization)
├── tests/          # testes pytest
├── reports/        # figuras, tabelas, PDFs e o relatório final em Word
├── docs/           # metodologia, dicionário, ADRs
└── scripts/        # pipelines de linha de comando
```

---

## 3. Pipeline geral

```
SEMAE (2009-2024)
   │
   ▼
[1] Carga e inspeção  →  [2] Pré-processamento  →  [3] Imputação
                                                       │
                                                       ▼
[6] Modelagem (K-Means, Random Forest, Isolation Forest)
   ▲                                                   │
   │                                                   ▼
[5] Detecção de outliers  ←  [4] Análise exploratória
   │
   ▼
[7] Validação (silhueta, F1, matriz de confusão)
   │
   ▼
[8] Relatório final (Word/PDF) + figuras
```

---

## 4. Passo a passo da metodologia em Python

### Etapa 1 — Carga e inspeção (`notebooks/01_carga_e_inspecao.ipynb`)

**Objetivo.** Carregar a base SEMAE, padronizar nomes de colunas e unidades, e gerar dicionário de variáveis.

**Como implementar.**
```python
import pandas as pd
from projeto_pcj.load import load_semae
from projeto_pcj.schema import SCHEMA

df = load_semae()  # lê data/interim/dados_organizados.csv; parser e tipagem padronizados
df.info()
df.describe().T
```

**Checklist.**
- Tipagem correta (datas como `datetime64`, numéricos como `float64`).
- Mapa de unidades publicado em `docs/dicionario_de_dados.md`.

---

### Etapa 2 — Pré-processamento (`notebooks/02_pre_processamento.ipynb`)

**Suposição declarada no projeto.** MCAR (Missing Completely At Random). O pipeline aplica o teste de Little (implementado em `src/preprocessing/missing.py`) para criticar a hipótese.

**Regras de decisão (do projeto):**
- Faltantes > 30% → excluir variável.
- Faltantes < 5% → imputar média (simétrica) ou mediana (assimétrica).
- 5–30% → imputar por KNN.
- Padronizar com `StandardScaler` antes do KNN sempre que faltantes > 15%.

**Validação da imputação.** Comparar média, mediana e desvio padrão antes/depois; verificar impacto no coeficiente de silhueta de um K-Means de referência.

---

### Etapa 3 — Imputação e retirada de outliers (`notebooks/03_imputacao.ipynb`, `05_outliers_iqr_vs_ensemble.ipynb`)

**Baseline: regra do IQR** vs. **ensemble** (KNN + Isolation Forest + LOF) por votação por maioria. O parâmetro `contamination` é calibrado por validação cruzada (estabilidade de Jaccard entre folds).

---

### Etapa 4 — Análise descritiva e exploratória (`notebooks/04_analise_exploratoria.ipynb`)

- Estatísticas descritivas (média, mediana, desvio padrão, quartis, assimetria, % faltantes) das bases bruta e imputada.
- Visualizações: histogramas, boxplots padronizados e séries temporais.
- Gerada de forma reprodutível por `scripts/run_descritiva.py`.

---

### Etapa 5 — Aplicação de modelos

- **K-Means** (`notebooks/06_kmeans_clustering.ipynb`): escolha de `k` por cotovelo + silhueta; perfis por cluster.
- **Random Forest** (`notebooks/07_random_forest.ipynb`): classificação de severidade (rotulagem CONAMA 357/2005 Classe 2, com salvaguarda anti-leakage).
- **Isolation Forest** (`notebooks/08_isolation_forest.ipynb`): detecção de anomalias (contamination 0,05–0,1).

---

### Etapa 6 — Análise das métricas (`notebooks/09_validacao_consolidada.ipynb`)

Reúne silhueta e WCSS do K-Means, F1/acurácia/matriz de confusão do Random Forest, taxa de anomalias do Isolation Forest e a tabela comparativa IQR × Ensemble.

---

### Etapa 7 — Relatório final (`notebooks/10_relatorio_final.ipynb`)

Gera as figuras e tabelas finais e o **relatório final em Word** (`reports/relatorio_final.docx`) por meio de `scripts/gerar_relatorio_docx.py`, além dos PDFs auxiliares.

---

## 5. Como rodar

Os scripts leem os caminhos de `src/config.py` (não recebem argumentos). A forma
mais simples é o **runner mestre**, que executa todas as etapas do cronograma na
ordem correta:

```bash
python scripts/run_all.py          # roda as etapas 2 a 7 do cronograma
python scripts/run_all.py --list   # lista as etapas
python scripts/run_all.py --only 5 6  # ex.: só modelos + relatório Word
```

As etapas do `run_all.py` correspondem a: `run_preprocess.py` →
`run_outliers.py` → `run_descritiva.py` → `run_train.py` → `run_evaluate.py` →
`gerar_relatorio_docx.py`.

Para rodar etapa por etapa:

```bash
# 0. Extração do PDF SEMAE → data/interim/dados_organizados.csv (rodar uma vez)
python scripts/extrair_dados.py

# 1. Pré-processamento e imputação (+ validação por silhueta do K-Means) + padronização
python scripts/run_preprocess.py

# 2. Análise de outliers completa (7 passos encadeados; usa matplotlib-venn)
python scripts/run_outliers.py             # roda tudo
python scripts/run_outliers.py --list      # lista os passos

# 3. Análise descritiva e exploratória (Etapa 4 do cronograma)
python scripts/run_descritiva.py   # descritiva_bruta.csv, descritiva_processada.csv, figuras eda_*.png

# 4. Modelagem: K-Means, Random Forest, Isolation Forest → models/, reports/tables/
python scripts/run_train.py

# 5. Métricas + figuras eval_*.png + relatorio_resultados.md + relatorio_final.pdf
python scripts/run_evaluate.py

# 6. RELATÓRIO FINAL em Word (~20 páginas) → reports/relatorio_final.docx
python scripts/gerar_relatorio_docx.py

# Relatórios em PDF auxiliares
python scripts/gerar_relatorio_pdf.py     # resumo auto (tabelas+figuras)
python scripts/relatorio_md_para_pdf.py   # relatório completo em PDF a partir do .md

# (Re)gerar os notebooks 00–10 a partir de src/
python scripts/gerar_notebooks.py

# Testes
pytest

# Ou, interativamente, executar os notebooks em ordem
jupyter lab notebooks/
```

**Deliverable principal:** `reports/relatorio_final.docx` — relatório de
resultados (~20 páginas, 16 figuras e 10 tabelas) gerado por
`scripts/gerar_relatorio_docx.py` a partir das tabelas e figuras do pipeline.

---

## 6. Reprodutibilidade

- **Sementes fixas** em todo modelo (`random_state=42`).
- **Versionamento de código** no GitHub.
- **Ambiente**: `requirements.txt` com versões mínimas fixadas.
- **Testes** cobrindo carga, imputação, outliers e métricas (`pytest`).
- **Documentação de decisões** em Architectural Decision Records (ADRs), em `docs/`.

---

## 7. Referências (selecionadas)

- Aggarwal, C. C. (2017). *Outlier Analysis*. Springer.
- Ayers, R. S.; Westcot, D. W. (1985). *Water quality for agriculture*. FAO 29.
- Biau, G.; Scornet, E. (2016). A random forest guided tour. *Test*, 25(2).
- Kaufman, L.; Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley.
- Little, R. J. A. (1988). A test of MCAR for multivariate data. *JASA*, 83(404).
- Liu, F. T. et al. (2008). Isolation Forest. *ICDM*.
- Pedregosa, F. et al. (2011). Scikit-learn. *JMLR*, 12.
- Powers, D. M. W. (2020). Evaluation: precision, recall, F-measure, ROC.
- Rousseeuw, P. J. (1987). Silhouettes. *J. Comp. Appl. Math.*, 20.
- Silva, R. F. et al. (2024). A Data-Driven Method for Water Quality Analysis and Prediction for Localized Irrigation. *AgriEngineering*, 6(2).
- Troyanskaya, O. et al. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics*, 17(6).
- van Buuren, S. (2018). *Flexible Imputation of Missing Data*. CRC Press.
- Zhao, Y. et al. (2019/2020). PyOD. *JMLR*.
- ANA (2021, 2024); PCJ (2024); ONU (2015, 2024); Brasil — CONAMA 357/2005.

> Lista completa: na seção "Referências" no projeto FAPESP original.
