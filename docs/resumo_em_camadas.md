# Resumo em camadas — Projeto FAPESP "Aplicação de Machine Learning para Análise da Qualidade da Água na Bacia do PCJ"

**Autores do projeto:** Guilherme Gramuglia Betta (aluno) e Profa. Dra. Patrícia Angélica Alves Marques (orientadora) — ESALQ/USP, em parceria com CENA-USP e C4AI.
**Período do cronograma:** novembro/2025 a outubro/2026.
**Fonte dos dados:** SEMAE Piracicaba — série histórica 2009–2024.

---

## 1. Estrutura de pastas do projeto Python (com notebooks de documentação)

A proposta abaixo segue boas práticas do "Cookiecutter Data Science", separando dados, código reutilizável (`src/`), notebooks de documentação narrativa (`notebooks/`), relatórios e testes.

```
projeto-pcj-ml/
├── README.md                         # Metodologia, instalação e como reproduzir
├── LICENSE
├── requirements.txt                  # Dependências fixadas (pip)
├── pyproject.toml                    # Configuração de build / lint / formatadores
├── .gitignore                        # Ignora /data, /venv, .ipynb_checkpoints, etc.
├── .pre-commit-config.yaml           # Hooks: black, ruff, nbstripout
│
├── data/                             # NUNCA versionar dados brutos no Git
│   ├── raw/                          # SEMAE 2009-2024 original (somente leitura)
│   ├── interim/                      # Dados pós-limpeza, antes da imputação
│   ├── processed/                    # Dados prontos para modelagem
│   └── external/                     # Limiares CONAMA 357/2005, mapas PCJ, etc.
│
├── notebooks/                        # Documentação executável (narrativa)
│   ├── 00_visao_geral.ipynb          # Apresentação do problema e objetivos
│   ├── 01_carga_e_inspecao.ipynb     # Leitura SEMAE, schema, dicionário de variáveis
│   ├── 02_pre_processamento.ipynb    # Tipos, unidades, % de faltantes, MCAR
│   ├── 03_imputacao.ipynb            # Média/mediana, KNN, comparação antes/depois
│   ├── 04_analise_exploratoria.ipynb # Estatísticas, histogramas, boxplots, correlação
│   ├── 05_outliers_iqr_vs_ensemble.ipynb # IQR vs. KNN + IsolationForest + LOF
│   ├── 06_kmeans_clustering.ipynb    # Cotovelo, silhueta, interpretação dos grupos
│   ├── 07_random_forest.ipynb        # Rotulagem CONAMA, treino, F1, matriz confusão
│   ├── 08_isolation_forest.ipynb     # Detecção de picos (P, N, turbidez)
│   ├── 09_validacao_consolidada.ipynb# Comparação final dos modelos
│   └── 10_relatorio_final.ipynb      # Síntese das figuras e tabelas do relatório
│
├── src/                              # Código reutilizável (importável)
│   ├── __init__.py
│   ├── config.py                     # Caminhos, sementes, limiares CONAMA
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py                   # Carga padronizada dos CSVs do SEMAE
│   │   └── schema.py                 # Dicionário de variáveis e unidades
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── missing.py                # MCAR check, KNN imputer, média/mediana
│   │   ├── scaling.py                # StandardScaler com persistência (joblib)
│   │   └── outliers.py               # IQR + ensemble (KNN, IForest, LOF)
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py         # Engenharia de variáveis e rotulagem CONAMA
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kmeans.py                 # Cotovelo + silhueta + persistência
│   │   ├── random_forest.py          # Treino com CV e tuning leve
│   │   └── isolation_forest.py       # Detecção de anomalias contextuais
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Silhueta, WCSS, F1, acurácia, matriz confusão
│   │   └── reports.py                # Geração automática de tabelas e PDFs
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                  # Histograma, boxplot, heatmap, scatter de clusters
│
├── tests/                            # Testes automatizados (pytest)
│   ├── test_load.py
│   ├── test_missing.py
│   ├── test_outliers.py
│   └── test_models.py
│
├── reports/
│   ├── figures/                      # PNG/SVG gerados pelos notebooks
│   ├── tables/                       # CSVs de métricas
│   ├── relatorio_parcial.pdf
│   └── relatorio_final.pdf
│
├── docs/
│   ├── metodologia.md                # Detalhamento das fórmulas (silhueta, F1, etc.)
│   ├── dicionario_de_dados.md        # Cada variável, unidade, faixa esperada, limite CONAMA
│   └── decisoes_de_projeto.md        # ADRs: imputação, contamination, k de clusters
│
└── scripts/                          # Pipelines reprodutíveis (linha de comando)
    ├── run_preprocess.py
    ├── run_train.py
    └── run_evaluate.py
```

**Boas práticas embutidas:**
- Notebooks numerados forçam ordem de leitura e reprodução.
- Toda lógica reutilizável vive em `src/`; notebooks apenas chamam funções e documentam decisões. Isso evita "código fantasma" perdido em células.
- Pasta `data/` versionada estruturalmente, mas dados ignorados pelo `.gitignore` (compatível com LGPD/uso responsável).
- `pre-commit` com `nbstripout` evita commitar outputs grandes; `black`/`ruff` mantêm o estilo.
- `tests/` cobre funções críticas (imputação, rotulagem CONAMA, métricas).
- `reports/` separa entregáveis (PDFs, figuras) do código.
- `scripts/` permite rodar o pipeline inteiro fora do Jupyter (essencial para reprodutibilidade real).

---

## 2. Resumo executivo

O projeto propõe usar técnicas de Machine Learning (ML) para organizar e analisar 15 anos de dados de qualidade de água (2009–2024) coletados pelo SEMAE no rio Piracicaba, com foco em apoiar decisões de manejo de irrigação na bacia PCJ. O conjunto inclui ~20 variáveis físico-químicas e biológicas (pH, turbidez, OD, nitrato, fósforo, condutividade elétrica, dureza, ferro, manganês, coliformes, cianobactérias, clorofila, etc.).

O argumento central é que monitorar qualidade pelos métodos tradicionais é caro e lento, e que ML pode (a) limpar e imputar a base, (b) detectar padrões espaço-temporais por agrupamento (K-Means), (c) classificar a água como adequada/inadequada para irrigação (Random Forest, usando como rótulos os limites da Resolução CONAMA 357/2005) e (d) sinalizar anomalias críticas (Isolation Forest). A entrega prática inclui notebooks reprodutíveis no GitHub, com pré-processamento robusto (tratamento MCAR, imputação por KNN, normalização StandardScaler) e validação tanto quantitativa (silhueta, WCSS, F1, acurácia, matriz de confusão) quanto visual.

O contexto é estratégico: a bacia PCJ abastece milhões de pessoas, transfere água para a RMSP via Sistema Cantareira, sustenta agroindústria de cana e um forte polo industrial de Campinas, e está sob pressão crescente de uso. O autor já foi bolsista FAPESP (2023/10336-8) e o projeto se conecta ao doutorado de Lucas Santiago Lima (CAPES). O cronograma é de 12 meses (nov/2025–out/2026), com possibilidade de BEPE.

---

## 3. Principais dados, argumentos e evidências citados

**Contexto agro-hídrico (Brasil/PCJ)**
- Brasil entre os 10 países com maior área irrigada: ~8,2 milhões de hectares, responsáveis por 16% da produção agropecuária nacional (ANA, 2021).
- Bacia PCJ: 76 municípios (71 SP + 5 MG), extensão de ~300 km; clima subtropical úmido; transposição pelo Sistema Cantareira para a RMSP (ANA, 2024; PCJ, 2024; Machado, Lopes & Duarte, 2025).
- Vínculo direto com o ODS 2 da ONU (fome zero e agricultura sustentável).

**Variáveis físico-químicas relevantes para irrigação**
- A literatura cita dezenas de parâmetros possíveis (alcalinidade, DBO, DQO, dureza, fosfato, IWQI, sólidos totais, SAR, salinidade, metais pesados, microrganismos etc.).
- Para irrigação especificamente: condutividade elétrica (CE), Fe, pH, dureza total, Na, Ca e microrganismos que causam entupimento de emissores (Ayers et al., 1985; Cordeiro, 2001; Muniz et al., 2022; Abadi et al., 2024).

**Impactos da má qualidade**
- pH desbalanceado altera solubilidade de nutrientes; nitrato em excesso induz crescimento vegetativo em detrimento da floração; metais pesados causam entupimentos físicos/químicos; sódio elevado degrada estrutura do solo e microbiota; salinidade danifica raízes e folhas.
- Entupimentos físicos/químicos/biológicos elevam custo de manutenção de irrigação (Cox et al., 2015; Muniz et al., 2022; Silva et al., 2024).

**Variáveis disponíveis na base SEMAE (2009–2024)**
Cor (ppm Pt Co), Turbidez (FTU), pH, Alcalinidade (ppm CaCO3), Acidez (ppm CaCO3), O.C. (ppm O2), DBO (ppm O2), Oxigênio Dissolvido (ppm O2), Cl⁻ (ppm Cl⁻), Dureza (ppm CaCO3), Fe (ppm Fe), Mn (mg/L), N (ppm N), P (ppm P), Condutividade Elétrica (µS/cm), Surfactantes (mg/L), Cianobactérias (células/mL), Coliformes Totais (NMP/100 mL), Coliformes Fecais (NMP/100 mL), Clorofila (µg/L), F (ppm F).

**Decisões metodológicas declaradas**
- Suposição de ausência: MCAR (Missing Completely At Random), comum em sensores hidrológicos (Lepot et al., 2017).
- Imputação: exclusão se >30% faltantes; média/mediana se <5%; KNN para 5–30%; padronização StandardScaler para percentuais >15% (Troyanskaya et al., 2001; Jordanov & Tarassov, 2019).
- Correlação por Pearson para detectar redundância (Hamzah et al., 2020).
- Outliers: ensemble KNN + Isolation Forest + Local Outlier Factor com `contamination` entre 0,05 e 0,1; comparação com IQR (Aggarwal, 2017; Zhao et al., 2020a,b).
- K-Means: `n_clusters` testado de 2 a 10 via método do cotovelo + silhueta (Kaufman & Rousseeuw, 1990; Rousseeuw, 1987).
- Random Forest: split 80/20, `n_estimators=100`, profundidade limitada, validação cruzada k=5; rótulos derivados da Resolução CONAMA 357/2005 (Biau & Scornet, 2016; Pedregosa et al., 2011).
- Isolation Forest: `contamination` 0,05–0,1 calibrado pela análise exploratória (Liu et al., 2008).

**Equações declaradas no projeto**
- Silhueta s(i) = (b(i) − a(i)) / max{a(i), b(i)}.
- WCSS = Σ Σ ||x − μᵢ||² (soma dos quadrados intra-cluster).
- Acurácia = (VP + VN) / (VP + VN + FP + FN).
- F1 = 2 · (Precisão · Recall) / (Precisão + Recall).

---

## 4. O que é realmente novo, útil ou acionável

- **Preenche uma lacuna regional.** O próprio texto reconhece que faltam métodos automáticos de avaliação de qualidade de água para irrigação especificamente para a bacia PCJ. Consolidar 15 anos do SEMAE num pipeline reprodutível já é, por si, um produto útil.
- **Comparação explícita IQR vs. ensemble de outliers.** Esse benchmark é raro em estudos aplicados de qualidade de água e pode virar uma publicação metodológica.
- **Rotulagem por norma legal (CONAMA 357/2005).** Atrelar a classificação supervisionada a um padrão regulatório torna o resultado diretamente acionável por gestores e operadores de irrigação.
- **Detecção de anomalias com Isolation Forest.** Apoia alertas operacionais para picos de P, N e turbidez que podem entupir emissores e comprometer cultivos.
- **Documentação em Jupyter + GitHub.** Cria base de reprodutibilidade que pode ser estendida a outras bacias e a outros parceiros (CENA-USP, C4AI).
- **Conexão com o doutorado em andamento** (Lucas Santiago Lima, CAPES) sugere continuidade e capilarização do conhecimento.

---

## 5. Pontos que exigem cautela, checagem ou contexto adicional

- **Suposição de MCAR.** Em dados de monitoramento ambiental, faltantes raramente são totalmente aleatórios — sensores costumam falhar mais em eventos extremos (justamente quando a informação mais importa). Vale testar MAR/MNAR e fazer análise de sensibilidade antes de fixar a estratégia de imputação.
- **Limiar de exclusão >30% e KNN 5–30%.** São heurísticas razoáveis mas arbitrárias. Convém justificar com curva de sensibilidade do impacto no coeficiente de silhueta e nas métricas do Random Forest.
- **Rotulagem CONAMA 357/2005 ≠ "adequado para irrigação".** A 357/2005 classifica corpos d'água por classes de uso; irrigação está incluída, mas faixas específicas de SAR, CE, B e Na para uso agrícola são tratadas em normas e guias adicionais (FAO 29 / Ayers & Westcot; Resolução CONAMA 396/2008 para águas subterrâneas; CONAMA 430/2011 para efluentes). O projeto cita uma "Resolução CONAMA 450/2012" referente a óleos lubrificantes (alteração da 362/2005) — isso parece um lapso de citação e precisa ser corrigido.
- **`contamination` entre 0,05 e 0,1 é uma escolha forte.** Implica supor que 5–10% dos pontos são anômalos, o que pode estar acima da taxa real em séries longas e bem operadas. Esse parâmetro deveria ser calibrado contra eventos conhecidos (laudos, ocorrências de poluição registradas).
- **Pearson assume relação linear.** Para variáveis assimétricas (turbidez, nitrato), considerar Spearman ou correlações condicionais.
- **Unidades inconsistentes na base.** "ppm O2" para DBO e OC, e "mg/L" para Mn no mesmo dataset; conferir se "O.C." é "Oxigênio Consumido" (DQO simplificada) ou outra variável. Padronizar antes de modelar.
- **Frequência amostral.** Não fica claro se é diário, semanal ou mensal — isso muda completamente o tratamento de sazonalidade e a detecção de picos. Vale verificar logo no início.
- **Risco de overfitting com Random Forest** se a rotulagem for derivada das próprias variáveis (ex.: classificar como "inadequada" usando a mesma turbidez que entra como feature). Definir cuidadosamente quais variáveis são features e quais são rótulos.
- **Cronograma apresenta inconsistências.** A Tabela 1 lista bimestres com "nov nov dez dez..." em 2025 e depois "maio maio jun jun..." em 2026 — provavelmente é representação visual com duas colunas por mês, mas convém revisar a formatação antes de submissão.
- **Citações repetidas no PDF.** A referência Aggarwal (2017) aparece duas vezes; Troyanskaya et al. (2021) parece estar com ano errado (o artigo original em Bioinformatics é de 2001).
- **Comparabilidade temporal.** 15 anos de dados podem ter mudanças de metodologia analítica no laboratório do SEMAE. Verificar mudanças de equipamento, limites de detecção e responsáveis técnicos.

---

## 6. Perguntas que você deveria fazer depois de ler este PDF

1. Qual a frequência exata de amostragem do SEMAE (diária? mensal?) e em quais pontos do rio?
2. Os dados são públicos? Há um termo de uso ou acordo de cooperação formal com o SEMAE?
3. Houve mudanças nos métodos analíticos do laboratório ao longo dos 15 anos? Como isso será controlado?
4. Por que escolher CONAMA 357/2005 como referência e não os critérios FAO 29 (Ayers & Westcot), que são especificamente para irrigação?
5. A "Resolução CONAMA 450/2012" citada (que trata de óleos lubrificantes) é mesmo a base regulatória pretendida ou houve confusão de citação?
6. Como será tratada a sazonalidade (chuva/seca, ciclos da cana) no clustering?
7. O Isolation Forest será aplicado por janela temporal (mês, trimestre) ou no dataset inteiro? Picos sazonais podem ser confundidos com anomalias.
8. Qual o plano para variáveis com >30% de faltantes — vão ser descartadas mesmo que sejam críticas (ex.: cianobactérias)?
9. Como será feita a rotulagem para o Random Forest? Por especialista, por limiar CONAMA, por consenso de múltiplos critérios?
10. Haverá comparação com algum baseline simples (regressão logística, árvore única) antes do Random Forest?
11. O modelo final será disponibilizado como API ou dashboard para o SEMAE / Comitê PCJ, ou apenas como repositório de pesquisa?
12. Como o projeto vai lidar com possíveis vieses de amostragem (locais de coleta concentrados em certos trechos)?
13. Há plano para incorporar dados externos (precipitação, uso do solo, vazão) que ajudariam a contextualizar anomalias?
14. Qual é o critério para considerar um cluster "interpretável"? Quem valida (limnólogo, agrônomo)?
15. Há comitê ou parceiro institucional que validará a utilidade prática dos alertas de anomalia?

---
