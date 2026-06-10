# Dicionário de Dados

Descreve as colunas do arquivo `data/interim/dados_organizados.csv`, gerado por
[`scripts/extrair_dados.py`](../scripts/extrair_dados.py) a partir de
[`data/raw/data_copy.pdf`](../data/raw/data_copy.pdf) (SEMAE-Piracicaba, série 2009–2024).

Cada linha do CSV representa **uma estatística** (`Mín.`, `Méd.` ou `Máx.`) de **um agregado mensal**
(`Jan`…`Dez`) ou **anual** (`Ano`) de **um ano** específico. Há 624 linhas: 16 anos × 13 grupos × 3 estatísticas.

---

## Colunas de identificação

| Coluna | Tipo | Domínio | Descrição |
|---|---|---|---|
| `Ano`  | int  | 2009–2024 | Ano de referência da página do PDF. |
| `Mes`  | str  | `Jan`, `Fev`, …, `Dez`, `Ano` | Agregado mensal ou resumo anual. `Ano` indica a linha de resumo do ano inteiro. |
| `Calc` | str  | `Mín.`, `Méd.`, `Máx.` | Estatística agregada (mínimo, média, máximo) no período. |

---

## Variáveis físico-químicas e bacteriológicas

Variáveis ordenadas como no CSV. Limites referenciais correspondem à **Resolução CONAMA 357/2005, Classe 2**
(águas doces, uso predominante incluindo irrigação de hortaliças e frutas; vide §5 sobre limitações).

| Coluna no CSV | Variável | Unidade no CSV | Unidade correta | Limite CONAMA 357/2005 (Classe 2) | Observações |
|---|---|---|---|---|---|
| `COR(ppm Pt Co)` | Cor verdadeira | ppm Pt-Co | mg Pt/L (equivalente a ppm Pt-Co) | ≤ 75 | Indicador de matéria orgânica dissolvida e ferro. |
| `TURB.(FTU)` | Turbidez | FTU | NTU (equivalente prático) | ≤ 100 | Material particulado em suspensão; correlaciona-se com chuva e erosão. |
| `pH` | Potencial hidrogeniônico | — | adimensional | 6,0 – 9,0 | Variável simétrica; usar média (não mediana) na imputação. |
| `ALC.(ppm CaCO3)` | Alcalinidade total | ppm CaCO₃ | mg CaCO₃/L | — (não normatizado) | Capacidade tampão; relevante para irrigação localizada (entupimento por carbonato). |
| `AC.(ppm CaCO3)` | Acidez total | ppm CaCO₃ | mg CaCO₃/L | — (não normatizado) | Acidez carbônica/mineral. |
| `O.C.(ppm O2)` | Oxigênio consumido (matéria orgânica via KMnO₄) | ppm O₂ | mg O₂/L | — | "Oxidabilidade"; substituto histórico para DQO. |
| `DBO(ppm O2)` | Demanda Bioquímica de Oxigênio (5 dias, 20 °C) | ppm O₂ | mg O₂/L | ≤ 5 | Muito esparsa na série SEMAE; valor `--` na maioria dos meses. |
| `O.D.(ppm O2)` | Oxigênio Dissolvido | ppm O₂ | mg O₂/L | ≥ 5 | Cair abaixo de 5 indica déficit (eutrofização, lançamento orgânico). |
| `Cl-(ppm Cl-)` | Cloreto | ppm Cl⁻ | mg Cl⁻/L | ≤ 250 | Indicador de salinidade; relevante para tolerância de cultivos. |
| `DUR.(ppm CaCO3)` | Dureza total (Ca²⁺ + Mg²⁺) | ppm CaCO₃ | mg CaCO₃/L | — (não normatizado) | Risco de incrustação em sistemas de irrigação. |
| `Fe(ppm Fe)` | Ferro total | ppm Fe | mg Fe/L | ≤ 0,3 (**dissolvido**) | A norma limita o Fe **dissolvido**, mas o SEMAE mede Fe **total** → comparar total contra o limite de dissolvido superestima violações. Em irrigação localizada, > 0,2 mg/L já obstrui emissores. |
| `Mn(mg/l)` | Manganês total | mg/L | mg/L | ≤ 0,1 (**total**) | O padrão CONAMA é "Manganês **total**" — coincide com a medição do SEMAE (sem problema de fração). Bioacumulação; reage com Fe formando precipitados. |
| `N(ppm N)` | Nitrogênio (forma total/Kjeldahl — verificar com SEMAE) | ppm N | mg N/L | NO₃⁻ ≤ 10; NO₂⁻ ≤ 1; N_amon. ≤ 3,7 (pH ≤ 7,5) | A coluna não distingue forma química; tratar como nitrogênio total agregado. Muito esparsa antes de 2018. |
| `P(ppm P)` | Fósforo total | ppm P | mg P/L | ≤ 0,1 (lóticos) | Indicador-chave de eutrofização. Muito esparso antes de 2018. |
| `Cond.(uS/cm)` | Condutividade elétrica | µS/cm | µS/cm | — (não normatizado) | Indicador integrado de sais dissolvidos; principal critério FAO 29 para irrigação. **No PDF aparece como `s/cm` (2009-2010, 2012-2024) ou `us/cm` (2011); o "µ" foi perdido na geração do PDF.** |
| `Amonia(mg/l)` | Amônia (N-amoniacal) | mg/L | mg N-NH₃/L | ≤ 3,7 (pH ≤ 7,5); ≤ 2,0 (7,5 < pH ≤ 8,0); … | **Variável só existe a partir de 2021** (preenchida como vazio para 2009-2020). |
| `Surfact.(mg/l)` | Surfactantes (substâncias reativas ao azul de metileno) | mg/L LAS | mg/L LAS | ≤ 0,5 | Detergentes; relacionado a esgoto sanitário. |
| `Cianobacteria(cel/ml)` | Cianobactérias | cél./mL | cél./mL | ≤ 50.000 | Risco de toxinas (microcistinas, saxitoxinas). No PDF de 2011 a coluna se chama `CIANOBAC-TÉRIAS` e fica no fim da tabela. |
| `S.T.(mg/l)` | Sólidos Totais | mg/L | mg/L | — (a 357/2005 normatiza sólidos *dissolvidos*: ≤ 500) | Soma de dissolvidos + suspensos. |
| `C.T.(NMP/100ml)` | Coliformes Totais | NMP/100 mL | NMP/100 mL | — (a 357/2005 normatiza termotolerantes; vide abaixo) | Indicador genérico de contaminação. |
| `C.F.(NMP/100ml)` | Coliformes Fecais / *E. coli* | NMP/100 mL | NMP/100 mL | ≤ 1.000 (irrigação de hortaliças cruas: ≤ 200) | **A partir de 2018 o SEMAE renomeou para `E COLI`; o script consolida ambas no mesmo campo.** |
| `CLOROFILA(ug/l)` | Clorofila-a | µg/L | µg/L | ≤ 30 | Proxy de biomassa fitoplanctônica. **No PDF de 2009-2010 e 2012-2024 a unidade aparece como `mg/l`, o que é fisicamente implausível — 2011 traz `ug/l`, confirmando que se trata de erro de rótulo nos demais anos.** |
| `F(ppm F)` | Fluoreto | ppm F⁻ | mg F⁻/L | ≤ 1,4 | **No PDF de 2009-2010 e 2012-2024 o cabeçalho diz `F (ppm N)`, o que é um erro tipográfico — 2011 traz `F- ppm F-`, confirmando que a variável é fluoreto em ppm F.** |

---

## Erratas conhecidas no PDF de origem (SEMAE)

Registradas aqui para corrigir a interpretação das colunas; o nome da coluna no CSV adota a forma **corrigida**.

1. **`F (ppm N)` → fluoreto em `ppm F`.** O cabeçalho de 2011 traz explicitamente `F- ppm F-`, comprovando que a coluna é flúor e que `ppm N` é uma errata propagada (provavelmente herança da coluna `N (ppm N)`).
2. **`CLOROFILA (mg/l)` → `µg/L`.** Valores típicos da série (1–50) são compatíveis apenas com µg/L; a página de 2011 já traz `ug /l` corretamente.
3. **`Cond. (s/cm)` → `µS/cm`.** O símbolo `µ` foi perdido na geração do PDF a partir de planilha; 2011 confirma com `us/cm`.

---

## Convenções e ausências

- **Separador decimal:** vírgula (`,`), como no PDF original. Converter para ponto no carregamento (`pd.read_csv(..., decimal=',')`).
- **Separador de milhar:** ponto (`.`), apenas em valores grandes (`1.650`, `2.400.000`). Tratar no parser.
- **Marcadores de ausência:** o PDF usa `--`, `---` ou `-` para indicar não-medido / abaixo do limite de detecção. O script preserva esses marcadores; a etapa de imputação ([`notebooks/03_imputacao.ipynb`](../notebooks/03_imputacao.ipynb)) deve mapeá-los para `NaN`.
- **DBO, N e P:** medidos com frequência muito baixa antes de 2018; provavelmente excederão o limite de 30% de faltantes definido em [`README.md`](../README.md) §4 Etapa 2 e serão descartados ou tratados em pipeline separado.

---

## Limitações da rotulagem baseada em CONAMA 357/2005

A 357/2005 classifica corpos d'água por uso, não por aptidão direta para irrigação. Para uso agrícola (especialmente
irrigação localizada), considerar também:

- **FAO 29 — Ayers & Westcot (1985):** restrições por condutividade elétrica, razão de adsorção de sódio (SAR), boro
  e bicarbonato. A base SEMAE não traz Na nem B, então alguns critérios FAO não são calculáveis com este dataset.
- **CONAMA 396/2008** (águas subterrâneas) e **CONAMA 430/2011** (efluentes) — fora do escopo direto desta base.
- A **"Resolução CONAMA 450/2012"** citada no PDF do projeto FAPESP é sobre óleos lubrificantes; trata-se de **erro de
  citação** e deve ser substituída por 357/2005 (e, idealmente, FAO 29) na rotulagem do Random Forest.
