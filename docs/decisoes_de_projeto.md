# Decisões de Projeto (ADRs)

Architectural Decision Records — registro curto das decisões metodológicas e de engenharia
que **não são óbvias pelo código** e que valem ser revisadas se algum pressuposto mudar.
Cada ADR segue o formato: contexto → decisão → consequências → status.

---

## ADR-001 — Extração do PDF SEMAE com `pdfplumber`

**Status:** Aceito (2026-05-13).
**Contexto.** O dado bruto chega como [`data/raw/data_copy.pdf`](../data/raw/data_copy.pdf) — 16 páginas
(uma por ano, 2009–2024) com tabelas físico-químicas. Não há CSV oficial. Alternativas avaliadas:

- `tabula-py` — depende do JRE Java, atrito de instalação no Windows.
- `pdfplumber` — Python puro, retorna a tabela já segmentada por linhas/células.
- `camelot` — exigiria Ghostscript no path.
- Conversão manual no Excel — não reprodutível.

**Decisão.** Usar `pdfplumber.extract_tables()` em [`scripts/extrair_dados.py`](../scripts/extrair_dados.py).
Mapeamento de cabeçalhos por **normalização** (`re.sub(r'[^a-z0-9]', '', s.lower())`) + dicionário, para
absorver as variações ano a ano (vide ADR-002). Mês é lido da **coluna 0 da linha `Mín.`** (primeira do trio),
estatística (`Mín./Méd./Máx.`) da coluna 1.

**Consequências.**
- Reprodutível e versionável (`pdfplumber>=0.11` em `requirements.txt`).
- Saída em `data/interim/dados_organizados.csv` segue a estrutura sugerida em
  [`docs/resumo_em_camadas.md`](resumo_em_camadas.md).
- Se o SEMAE mudar o layout do PDF, é provável que apenas o `HEADER_MAP` em
  `scripts/extrair_dados.py` precise ser atualizado.

---

## ADR-002 — Tratamento das variações de cabeçalho entre 2009 e 2024

**Status:** Aceito (2026-05-13).
**Contexto.** O cabeçalho do PDF muda ao longo dos anos:

| Período | Particularidade |
|---|---|
| 2009-2010, 2012-2017 | Layout "padrão" (22 variáveis nomeadas, 24 colunas). |
| 2011 | Layout reordenado, traz `Fenol` extra, `P ppm P`, `F- ppm F-`, `Cond. us/cm`, `CIANOBAC-TÉRIAS` no fim. |
| 2018-2020 | `C.F.` renomeada para `E COLI` (mesma variável). |
| 2021-2024 | Coluna `Amônia` adicionada entre `Cond.` e `Surfact.` (25 colunas). |

**Decisão.**
1. **Amônia entra no schema canônico** (23ª variável) e fica vazia para 2009-2020.
2. `E COLI` é consolidada na mesma coluna que `C.F.(NMP/100ml)` (são equivalentes).
3. `Fenol` (existe só em 2011) é **descartado** — fora do escopo, baixa cobertura.
4. **Três erratas do PDF de origem são corrigidas no nome da coluna do CSV** (com nota em
   [`docs/dicionario_de_dados.md`](dicionario_de_dados.md)):
   - `F (ppm N)` → `F(ppm F)` (em 2011 vem `F- ppm F-`, confirmando fluoreto).
   - `CLOROFILA (mg/l)` → `CLOROFILA(ug/l)` (em 2011 vem `ug /l`; mg/L é fisicamente implausível para clorofila-a).
   - `Cond. (s/cm)` → `Cond.(uS/cm)` (o caractere `µ` foi perdido na geração do PDF).

**Consequências.**
- `Amonia(mg/l)` terá ~75% de faltantes globais (medida só ≥ 2021). Pela regra do README (§4 Etapa 2,
  >30% → excluir), tenderá a ser descartada — vide ADR-004.
- A unificação `E COLI = C.F.` exige justificativa metodológica no relatório
  (são proxies aceitos, mas `E. coli` é subconjunto estrito de termotolerantes).
- Quem ler só o CSV (sem o dicionário) **não saberá** das erratas; por isso o cabeçalho está corrigido
  e o dicionário documenta a forma original do PDF.

---

## ADR-003 — Parser numérico brasileiro tolerante a formatos mistos

**Status:** Aceito (2026-05-13).
**Contexto.** O CSV intermediário preserva a notação do PDF: vírgula como decimal e ponto como milhar.
A coluna `C.F.(NMP/100ml)` apresenta **formatos mistos** no mesmo arquivo:
`'23.000'` (inteiro, 23 mil), `'30.000,00'` (decimal explícito), `'2.400.000'` (milhão).
O atalho `pd.read_csv(decimal=',', thousands='.')` falha porque pandas não consegue inferir
o tipo da coluna quando há ambiguidade `'23.000'` (23 ou 23000?).

**Decisão.** Em [`src/projeto_pcj/load.py`](../src/projeto_pcj/load.py), ler tudo como string e converter
com `_parse_br_number`:
- Se há vírgula → pontos são milhar, vírgula é decimal.
- Sem vírgula, só com pontos → milhar **somente** se o padrão for `\d{1,3}(\.\d{3})+` (caso de
  `23.000`, `1.300.000`); caso contrário trata como decimal nativo (`7.2`).
- Marcadores `--`, `---`, `-` viram `NaN`.

**Consequências.** Função única, testável, com fallback explícito para `NaN`. Toda a entrada numérica
no pipeline passa por ela — qualquer mudança de convenção pelo SEMAE é resolvida em um único lugar.

---

## ADR-004 — Estratégia de imputação por faixa de faltantes

**Status:** Provisório — segue o que o projeto FAPESP declara; pode mudar após a análise de sensibilidade.
**Contexto.** Análise inicial (cobertura temporal via `cobertura_temporal()`) mostra:

| Variável | % faltantes global | Observação |
|---|---:|---|
| `S.T.(mg/l)` | 99,0 | Quase nunca medida no boletim mensal. |
| `Surfact.(mg/l)` | 99,0 | Idem. |
| `DBO(ppm O2)` | 95,5 | Medidas esparsas só em alguns anos. |
| `P(ppm P)` | 89,4 | Concentrada em 2011 e 2022-2024. |
| `N(ppm N)` | 88,0 | Idem. |
| `Amonia(mg/l)` | 75,5 | Variável só existe ≥ 2021. |
| `C.T.(NMP/100ml)` | 60,6 | — |
| `C.F.(NMP/100ml)` | 18,4 | Variável mais bem medida do grupo bacteriológico. |

**Decisão (regras do projeto, herdadas do README §4 Etapa 2 e Lepot et al. 2017):**
- `> 30%` faltantes → **excluir variável**. Aplica-se a `S.T.`, `Surfact.`, `DBO`, `P`, `N`, `Amonia`, `C.T.` —
  ou seja, sete das 23 acabariam fora.
- `5% – 30%` → imputar por **KNN** com `n_neighbors=5`, `weights="distance"`, após `StandardScaler`.
- `< 5%` → imputar **média** (simétrica) ou **mediana** (assimétrica: turbidez, nitrato).
- Suposição declarada: **MCAR** (Lepot et al., 2017). Há crítica em
  [`docs/resumo_em_camadas.md`](resumo_em_camadas.md) §5.

**Consequências.** A regra dos 30% **descartaria nutrientes (N, P) e Amônia**, que são justamente as
variáveis mais relevantes para o objetivo agrícola. Isso é um conflito sério com o objetivo do projeto.
**Plano de mitigação** antes de fixar a regra:

1. Testar a hipótese MCAR com o teste de Little (`statsmodels`/`pingouin`).
2. Análise de sensibilidade: rodar o pipeline mantendo N, P e Amônia (e com KNN agressivo) e comparar
   silhueta do K-Means e F1 do Random Forest contra a versão "limpa".
3. Considerar split temporal: pipeline 1 = 2018-2024 com nutrientes; pipeline 2 = 2009-2024 sem.

---

## ADR-005 — Base legal para rotulagem (Random Forest)

**Status:** Pendente de validação com a orientadora.
**Contexto.** O projeto FAPESP cita **"CONAMA 450/2012"** como base de rotulagem; esse normativo
trata de **óleos lubrificantes** (altera a 362/2005) e **não se aplica** à qualidade da água do PCJ.
O substituto natural seria **CONAMA 357/2005, Classe 2**, mas a Classe 2 não é específica para irrigação
(inclui também recreação primária e dessedentação animal).

**Decisão preliminar.**
- Rotular adequação **multicritério**:
  - **Camada A:** CONAMA 357/2005 Classe 2 (limites em
    [`src/projeto_pcj/schema.py`](../src/projeto_pcj/schema.py)).
  - **Camada B:** FAO 29 — Ayers & Westcot (1985), para uso agrícola (CE, SAR, B).
    SAR não é calculável com a base atual (faltam Na, Ca, Mg dissolvidos) — registrar como limitação.
- Marcar "adequada/inadequada" como **intersecção** das duas camadas (mais conservador) ou
  **união** das violações (mais sensível). Decisão final após primeiro experimento.

**Consequências.** Definir a operação de junção é uma decisão de validade externa — pode mudar a métrica
F1 reportada em ordens de grandeza. Registrar a operação escolhida no relatório final.

**Risco de vazamento (data leakage).** Se um rótulo for derivado de uma variável que entra como feature
(ex.: rotular como "inadequada" usando `TURB.` e ainda passar `TURB.` para o modelo), as métricas ficam
artificialmente altas. **Regra:** variável usada na regra de rotulagem **não pode** ser feature do classificador.

---

## ADR-006 — Hiperparâmetros dos modelos (sementes fixas)

**Status:** Aceito (valores iniciais; sujeitos a tuning com GridSearchCV).

| Modelo | Hiperparâmetros iniciais | Justificativa |
|---|---|---|
| K-Means | `n_clusters=2..10`, `n_init="auto"`, `random_state=42` | Cotovelo + silhueta (Rousseeuw 1987; Kaufman & Rousseeuw 1990). Silhueta > 0,5 = clusters efetivos. |
| Random Forest | `n_estimators=100`, `max_depth=10`, `random_state=42`, `n_jobs=-1`, `cv=5` | Padrão Biau & Scornet 2016; `max_depth=10` limita overfit em base pequena (624 linhas). |
| Isolation Forest | `contamination=0,07`, `n_estimators=200`, `random_state=42` | Faixa 0,05-0,1 citada em Liu et al. 2008 — vide ADR-007. |
| Local Outlier Factor | `contamination=0,07`, `novelty=True` | Compatibilidade com ensemble. |
| PyOD KNN | `contamination=0,07` | Idem. |
| StandardScaler | aplicado **antes** do KNN se faltantes > 15% | Recomendação do projeto FAPESP. |

**Consequência crítica.** `random_state=42` em **todos** os modelos garante reprodutibilidade exata, mas
**não** garante robustez do resultado. Para validar: rodar com pelo menos 10 sementes diferentes e
reportar média ± desvio das métricas (em [`notebooks/09_validacao_consolidada.ipynb`](../notebooks/09_validacao_consolidada.ipynb)).

---

## ADR-007 — `contamination=0,07` no ensemble de outliers e Isolation Forest

**Status:** Provisório — calibrar contra eventos conhecidos.
**Contexto.** `contamination` é o **percentual prévio assumido de anomalias** na base.
Fixar em 0,07 implica supor que ~7% dos pontos são anômalos — equivalente a ~44 das 624 linhas.

**Decisão preliminar.** Manter 0,07 como ponto de partida (centro da faixa 0,05-0,1 do projeto FAPESP),
mas **calibrar** em [`notebooks/05_outliers_iqr_vs_ensemble.ipynb`](../notebooks/05_outliers_iqr_vs_ensemble.ipynb):

1. Listar eventos hidrológicos/poluentes conhecidos no período (laudos do SEMAE, notícias de
   contaminação, ocorrências da CETESB) — **se possível** com o SEMAE como fonte primária.
2. Variar `contamination` em {0,03, 0,05, 0,07, 0,10, 0,15} e medir **recall** dos eventos conhecidos.
3. Reportar a curva e fixar o valor que maximiza F1 contra a lista de referência.

**Consequência.** Sem calibração externa, 0,07 é uma escolha arbitrária — anomalias verdadeiras podem
ser perdidas ou falsos positivos podem dominar. Janela temporal (mês/trimestre) deve ser considerada
para não confundir sazonalidade com anomalia.

---

## ADR-008 — Versionamento de dados via hash + `.gitignore`

**Status:** Aceito.
**Contexto.** A pasta `data/` está no `.gitignore` (dados brutos não vão para o GitHub) por motivos de
licença e tamanho. Ainda assim, é preciso garantir que dois pesquisadores executando o pipeline em
máquinas diferentes partam do **mesmo PDF**.

**Decisão.**
- Registrar o **SHA-256** do arquivo `data/raw/data_copy.pdf` neste documento sempre que o PDF for
  atualizado. O hash atual deve ser regenerado pelo aluno e adicionado abaixo.
- Mesma regra para o CSV intermediário (`data/interim/dados_organizados.csv`).
- Comando: `python -c "import hashlib; print(hashlib.sha256(open('data/raw/data_copy.pdf','rb').read()).hexdigest())"`

| Arquivo | SHA-256 | Data |
|---|---|---|
| `data/raw/data_copy.pdf` | `f95d56080379d99d3b263de8fb5389abede56c8f89bb4e55276364c4c7173aa2` | 2026-05-13 |
| `data/interim/dados_organizados.csv` | `f35921ac4c89b780bbd212b78fb4fe6c8ab75efb32eaa13381126f8ed7c8e460` | 2026-05-13 |

**Consequências.** Reprodutibilidade verificável sem versionar dados sensíveis.
