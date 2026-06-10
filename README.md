# Aplicação de Machine Learning para Análise da Qualidade da Água na Bacia do PCJ

Projeto de Iniciação Científica (FAPESP) — ESALQ/USP
Aluno: Guilherme Gramuglia Betta · Orientadora: Profa. Dra. Patrícia Angélica Alves Marques
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
cd projeto-pcj-ml

# Criar e ativar ambiente virtual
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
# (opcional, para desenvolvimento)
pip install -r requirements-dev.txt
pre-commit install
```

`requirements.txt` mínimo:

```
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
scipy>=1.13
matplotlib>=3.8
seaborn>=0.13
pyod>=2.0
joblib>=1.4
jupyterlab>=4.2
ipykernel>=6.29
missingno>=0.5
```

---

## 2. Estrutura do repositório

Veja `resumo_em_camadas.md` (seção 1) ou a árvore (abaixo). Notebooks numerados em `notebooks/` reproduzem o pipeline; o código reutilizável vive em `src/projeto_pcj/`.

```
projeto-pcj-ml/
├── data/           # raw, interim, processed, external (não versionado)
├── notebooks/      # 00..10 — passo a passo executável
├── src/projeto_pcj # módulos Python (data, preprocessing, features, models, evaluation)
├── tests/          # testes pytest
├── reports/        # figuras, tabelas, PDFs
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
[8] Relatório final + figuras
```

---

## 4. Passo a passo da metodologia em Python

### Etapa 1 — Carga e inspeção (`notebooks/01_carga_e_inspecao.ipynb`)

**Objetivo.** Carregar a base SEMAE, padronizar nomes de colunas e unidades, e gerar dicionário de variáveis.

**Variáveis esperadas (21):** Cor (ppm Pt Co), Turbidez (FTU), pH, Alcalinidade (ppm CaCO3), Acidez (ppm CaCO3), O.C. (ppm O2), DBO (ppm O2), Oxigênio Dissolvido (ppm O2), Cl⁻ (ppm Cl⁻), Dureza (ppm CaCO3), Fe (ppm Fe), Mn (mg/L), N (ppm N), P (ppm P), Condutividade Elétrica (µS/cm), Surfactantes (mg/L), Cianobactérias (céls/mL), Coliformes Totais (NMP/100 mL), Coliformes Fecais (NMP/100 mL), Clorofila (µg/L), F (ppm F).

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
- Verificação de duplicatas e de pontos de coleta (se houver).

---

### Etapa 2 — Pré-processamento (`notebooks/02_pre_processamento.ipynb`)

**Análise de faltantes.**
```python
import missingno as msno
msno.matrix(df)
faltantes = df.isna().mean().sort_values(ascending=False)
```

**Suposição declarada no projeto.** MCAR (Missing Completely At Random), seguindo Lepot et al. (2017). **Recomendação:** testar a hipótese com o teste de Little (`statsmodels` ou `pingouin`) e fazer análise de sensibilidade.

**Regras de decisão (do projeto):**
- Faltantes > 30% → excluir variável.
- Faltantes < 5% → imputar média (simétrica) ou mediana (assimétrica como turbidez e nitrato).
- 5–30% → imputar por KNN.
- Padronizar com `StandardScaler` antes do KNN sempre que faltantes > 15%.

**Implementação.**
```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
imputer = KNNImputer(n_neighbors=5, weights="distance")
X_imputed = imputer.fit_transform(X_scaled)
```

**Validação da imputação.** Comparar média, mediana e desvio padrão antes/depois; verificar impacto no coeficiente de silhueta de um K-Means de referência.

**Correlação.**
```python
import seaborn as sns
corr = df[features].corr(method="pearson")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
```
Considerar também Spearman para variáveis assimétricas.

---

### Etapa 3 — Análise exploratória (`notebooks/04_analise_exploratoria.ipynb`)

- Estatísticas descritivas (média, mediana, desvio padrão, quartis).
- Visualizações: histograma, boxplot por variável, série temporal por variável, heatmap de correlação.
- Foco em pH, turbidez, nitrato, fósforo e condutividade elétrica.
- Sinalizar variáveis com distribuição muito assimétrica (candidatas a transformação log).

---

### Etapa 4 — Detecção de outliers (`notebooks/05_outliers_iqr_vs_ensemble.ipynb`)

**Baseline: regra do IQR.**
```python
q1, q3 = df[col].quantile([0.25, 0.75])
iqr = q3 - q1
mask = (df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)
```

**Ensemble (KNN + Isolation Forest + Local Outlier Factor).**
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.knn import KNN

iforest = IsolationForest(contamination=0.07, random_state=42)
lof = LocalOutlierFactor(contamination=0.07, novelty=True)
knn = KNN(contamination=0.07)

# Votação por maioria
votes = (iforest.fit_predict(X) == -1).astype(int) + \
        (lof.fit(X).predict(X) == -1).astype(int) + \
        knn.fit(X).predict(X)
outliers = votes >= 2
```

**Sugestão de melhoria.** Calibrar `contamination` em validação cruzada (Zhao et al., 2020a,b) e comparar com IQR em termos de robustez, especialmente para picos de nitrato e turbidez.

---

### Etapa 5 — K-Means (clustering) — `notebooks/06_kmeans_clustering.ipynb`

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wcss, sil = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
```

- Escolher `k` pelo método do cotovelo (Kaufman & Rousseeuw, 1990) e pela silhueta (Rousseeuw, 1987) — valores > 0,5 indicam clusters efetivos.
- Interpretar clusters cruzando médias por variável e sazonalidade.
- Validação visual: scatter plots (ex.: pH × turbidez) coloridos por cluster.

---

### Etapa 6 — Random Forest (classificação) — `notebooks/07_random_forest.ipynb`

**Rotulagem.** Definir "adequada/inadequada" com base em padrões legais (citados no projeto como CONAMA 357/2005). **Atenção:** o projeto cita também "CONAMA 450/2012", que trata de óleos lubrificantes — confirmar a base legal correta antes da rotulagem. Para irrigação, considerar também FAO 29 (Ayers & Westcot).

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

- Métricas: acurácia e F1 (Biau & Scornet, 2016; Powers, 2020).
- Matriz de confusão para destacar falsos positivos/negativos (Congalton, 1991).
- Avaliar importância de variáveis (`clf.feature_importances_`) para conectar com a literatura.

---

### Etapa 7 — Isolation Forest (anomalias) — `notebooks/08_isolation_forest.ipynb`

```python
from sklearn.ensemble import IsolationForest

iforest = IsolationForest(contamination=0.07, n_estimators=200, random_state=42)
df["anomalia"] = iforest.fit_predict(X) == -1
```

- Ajustar `contamination` 0,05–0,1 com base na análise exploratória (Liu et al., 2008).
- Considerar janelas temporais (mês ou trimestre) para evitar confundir sazonalidade com anomalia.
- Validação visual: scatter plots destacando os pontos anômalos (P, N e turbidez são prioridade).

---

### Etapa 8 — Validação consolidada (`notebooks/09_validacao_consolidada.ipynb`)

Reúne em um único dashboard:
- Silhueta e WCSS do K-Means.
- F1, acurácia e matriz de confusão do Random Forest.
- Taxa de anomalias do Isolation Forest, com confronto contra eventos registrados.
- Tabela comparativa IQR × Ensemble × Modelos supervisionados.

---

### Etapa 9 — Relatório final (`notebooks/10_relatorio_final.ipynb`)

Gera as figuras e tabelas finais (em `reports/figures/` e `reports/tables/`), exporta o `relatorio_final.pdf` e cria a versão "executiva" do estudo.

---

## 5. Como rodar

Os scripts leem os caminhos de `src/config.py` (não recebem argumentos). Ordem recomendada:

```bash
# 0. Extração do PDF SEMAE → data/interim/dados_organizados.csv (rodar uma vez)
python scripts/extrair_dados.py

# 1. Pipeline de ML (pré-processamento → modelagem → avaliação)
python scripts/run_preprocess.py   # imputação (+ validação por silhueta do K-Means) + padronização
python scripts/run_train.py        # K-Means, Random Forest, Isolation Forest → models/, reports/tables/
python scripts/run_evaluate.py     # métricas + figuras eval_*.png + relatorio_resultados.md + relatorio_final.pdf

# 2. Análise de outliers completa (6 passos encadeados)
python scripts/run_outliers.py             # roda tudo
python scripts/run_outliers.py --list      # lista os passos
python scripts/run_outliers.py --only 4 5 6  # só tabelas + figuras novas

# 3. Relatório final em PDF (também gerado automaticamente por run_evaluate.py)
python scripts/gerar_relatorio_pdf.py   # → reports/relatorio_final.pdf

# 4. (Re)gerar os notebooks 00–10 a partir de src/
python scripts/gerar_notebooks.py

# 5. Testes
pytest

# Ou, interativamente, executar os notebooks em ordem
jupyter lab notebooks/
```

---

## 6. Reprodutibilidade

- **Sementes fixas** em todo modelo (`random_state=42`).
- **Versionamento de código** no GitHub.
- **Versionamento de dados**: hash SHA-256 do CSV bruto registrado em `docs/decisoes_de_projeto.md`.
- **Notebooks limpos** antes de commitar (`pre-commit` + `nbstripout`).
- **Ambiente**: `requirements.txt` com versões pinadas; opcionalmente `Dockerfile` para isolamento total.
- **Testes** mínimos cobrindo carga, imputação, rotulagem e métricas.
- **Documentação de decisões** em Architectural Decision Records (ADRs).

---

## 7. Referências (selecionadas)

- Aggarwal, C. C. (2017). *Outlier Analysis*. Springer.
- Biau, G.; Scornet, E. (2016). A random forest guided tour. *Test*, 25(2).
- Kaufman, L.; Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley.
- Lepot, M. et al. (2017). Interpolation in time series. *Water*, 9(10), 796.
- Liu, F. T. et al. (2008). Isolation Forest.
- Pedregosa, F. et al. (2011). Scikit-learn. *JMLR*, 12.
- Powers, D. M. W. (2020). Evaluation: precision, recall, F-measure, ROC.
- Rousseeuw, P. J. (1987). Silhouettes. *J. Comp. Appl. Math.*, 20.
- Silva, R. F. et al. (2024). A Data-Driven Method for Water Quality Analysis and Prediction for Localized Irrigation. *AgriEngineering*, 6(2).
- Troyanskaya, O. et al. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics*, 17(6).
- Zhao, Y. et al. (2020a, 2020b). PyOD / COPOD.
- ANA (2021, 2024); PCJ (2024); ONU (2015, 2024); Brasil — CONAMA 357/2005.

> Lista completa: na seção "Referências" no projeto FAPESP original.
