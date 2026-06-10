"""Extrai os dados tabulares do PDF do SEMAE (Água Bruta - rio Piracicaba) para CSV.

O PDF de entrada (data/raw/data_copy.pdf) tem 16 páginas, uma por ano (2009-2024).
Em cada página há uma tabela com 22-23 variáveis físico-químicas/bacteriológicas,
três linhas por mês (Mín./Méd./Máx.) e três linhas de resumo anual.

Particularidades tratadas:
    * 2011 reordena colunas e troca alguns rótulos (P ppm P, F- ppm F-, Cond. us/cm,
      "Fenol" extra que é ignorado, CIANOBAC-TÉRIAS no fim).
    * 2018-2020 renomeiam "C.F." para "E COLI" (mesma variável).
    * 2021-2024 adicionam a coluna "Amônia" entre Cond. e Surfact.

Saída: CSV "longo" em data/interim/dados_organizados.csv com colunas
    Ano, Mes, Calc, <variáveis...>
onde Mes ∈ {Jan..Dez, Ano} e Calc ∈ {Mín., Méd., Máx.}.

Uso:
    python scripts/extrair_dados.py
    python scripts/extrair_dados.py --pdf data/raw/data_copy.pdf --out data/interim/dados_organizados.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import pdfplumber

LOG = logging.getLogger("extrair_dados")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "data" / "raw" / "data_copy.pdf"
DEFAULT_OUT = PROJECT_ROOT / "data" / "interim" / "dados_organizados.csv"

# Ordem fixa das colunas de saída. Inclui Amônia (existe a partir de 2021).
TARGET_COLUMNS: list[str] = [
    "COR(ppm Pt Co)", "TURB.(FTU)", "pH", "ALC.(ppm CaCO3)", "AC.(ppm CaCO3)",
    "O.C.(ppm O2)", "DBO(ppm O2)", "O.D.(ppm O2)", "Cl-(ppm Cl-)", "DUR.(ppm CaCO3)",
    "Fe(ppm Fe)", "Mn(mg/l)", "N(ppm N)", "P(ppm P)", "Cond.(uS/cm)", "Amonia(mg/l)",
    "Surfact.(mg/l)", "Cianobacteria(cel/ml)", "S.T.(mg/l)", "C.T.(NMP/100ml)",
    "C.F.(NMP/100ml)", "CLOROFILA(ug/l)", "F(ppm F)",
]

# Cabeçalho normalizado (alfanuméricos minúsculos) -> coluna destino.
# Cobre as variações de grafia observadas entre 2009 e 2024.
HEADER_MAP: dict[str, str] = {
    "corppmptco": "COR(ppm Pt Co)",
    "turbftu": "TURB.(FTU)",
    "ph": "pH",
    "alcppmcaco3": "ALC.(ppm CaCO3)",
    "acppmcaco3": "AC.(ppm CaCO3)",
    "ocppmo2": "O.C.(ppm O2)",
    "dboppmo2": "DBO(ppm O2)",
    "odppmo2": "O.D.(ppm O2)",
    "clppmcl": "Cl-(ppm Cl-)",
    "durppmcaco3": "DUR.(ppm CaCO3)",
    "feppmfe": "Fe(ppm Fe)",
    "mnmgl": "Mn(mg/l)",
    "nppmn": "N(ppm N)",
    "pppmp": "P(ppm P)",
    "condscm": "Cond.(uS/cm)",
    "conduscm": "Cond.(uS/cm)",
    "amnia": "Amonia(mg/l)",
    "amonia": "Amonia(mg/l)",
    "surfactmgl": "Surfact.(mg/l)",
    "cianobacteriancelml": "Cianobacteria(cel/ml)",
    "cianobactriasnclml": "Cianobacteria(cel/ml)",
    "stmgl": "S.T.(mg/l)",
    "ctnmp100ml": "C.T.(NMP/100ml)",
    "cfnmp100ml": "C.F.(NMP/100ml)",
    "ecolinmp100ml": "C.F.(NMP/100ml)",
    "clorofilamgl": "CLOROFILA(ug/l)",
    "clorofilaugl": "CLOROFILA(ug/l)",
    "fppmn": "F(ppm F)",
    "fppmf": "F(ppm F)",
}

MESES = {"Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"}

# Mapeia estatística (com ou sem acento, encoding ruim do PDF) para forma canônica.
STAT_MAP = {
    "min.": "Mín.", "mn.": "Mín.", "mín.": "Mín.",
    "med.": "Méd.", "md.": "Méd.", "méd.": "Méd.",
    "max.": "Máx.", "mx.": "Máx.", "máx.": "Máx.",
}


def _norm(s: str | None) -> str:
    """Minúsculas + apenas alfanuméricos. Robusto a espaços, acentos e encoding ruim."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _stat(cell: str | None) -> str | None:
    """Identifica Mín./Méd./Máx. tolerando encoding latin-1 corrompido do PDF."""
    if not cell:
        return None
    key = cell.strip().lower().replace("ï¿½", "")  # remove '�' fragments
    key = re.sub(r"[^a-z.]", "", key)
    return STAT_MAP.get(key)


def _ano_da_pagina(table: list[list]) -> str | None:
    """Procura 'Ano: YYYY' no título da tabela (primeira linha não vazia)."""
    for row in table[:3]:
        for cell in row:
            if cell and "Ano:" in cell:
                m = re.search(r"Ano:\s*(\d{4})", cell)
                if m:
                    return m.group(1)
    return None


def _linha_de_cabecalho(table: list[list]) -> tuple[int, list[str]] | None:
    """Retorna (índice, linha) do cabeçalho — linha em que aparece 'COR'."""
    for idx, row in enumerate(table):
        if any(c and "COR" in str(c) for c in row):
            return idx, [str(c) if c is not None else "" for c in row]
    return None


def _mapear_colunas(header: list[str]) -> dict[str, int]:
    """Mapeia coluna destino -> índice da coluna no PDF, via HEADER_MAP."""
    mapping: dict[str, int] = {}
    for idx, cell in enumerate(header):
        # Cabeçalho do PDF vem em duas linhas; achata e normaliza
        norm = _norm(cell.replace("\n", " "))
        if not norm:
            continue
        if norm in HEADER_MAP:
            destino = HEADER_MAP[norm]
            # Em caso de duplicata (improvável), mantém o primeiro
            mapping.setdefault(destino, idx)
    return mapping


def _processar_pagina(page) -> list[list[str]]:
    """Extrai linhas (Ano, Mes, Calc, *valores) de uma página."""
    tables = page.extract_tables()
    if not tables:
        return []
    table = tables[0]

    ano = _ano_da_pagina(table)
    if not ano:
        LOG.warning("Página sem 'Ano: YYYY' identificável - ignorada")
        return []

    cab = _linha_de_cabecalho(table)
    if not cab:
        LOG.warning("Ano %s: cabeçalho não localizado - ignorado", ano)
        return []
    cab_idx, cab_linha = cab
    pos = _mapear_colunas(cab_linha)
    LOG.info("Ano %s: %d/%d colunas mapeadas", ano, len(pos), len(TARGET_COLUMNS))

    linhas_saida: list[list[str]] = []
    mes_atual: str | None = None

    for row in table[cab_idx + 1 :]:
        if not row or all(c is None or not str(c).strip() for c in row):
            continue

        col0 = (row[0] or "").strip()
        # Atualiza o mês "ativo" quando a primeira coluna traz um nome de mês ou 'Ano'
        if col0 == "Ano":
            mes_atual = "Ano"
        elif col0 in MESES:
            mes_atual = col0

        stat = _stat(row[1] if len(row) > 1 else None)
        if not stat or not mes_atual:
            # Linha que não é Mín./Méd./Máx. (cabeçalho-fantasma, separador etc.)
            continue

        linha_out: list[str] = [ano, mes_atual, stat]
        for col in TARGET_COLUMNS:
            idx = pos.get(col)
            if idx is None or idx >= len(row):
                linha_out.append("")
            else:
                valor = row[idx]
                linha_out.append(("" if valor is None else str(valor)).strip())
        linhas_saida.append(linha_out)

    return linhas_saida


def processar_pdf(pdf_path: Path, csv_path: Path) -> int:
    """Lê o PDF e escreve o CSV de saída. Retorna a quantidade de linhas escritas."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF de entrada não encontrado: {pdf_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    todas: list[list[str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            LOG.debug("Processando página %d/%d", i, len(pdf.pages))
            todas.extend(_processar_pagina(page))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(["Ano", "Mes", "Calc"] + TARGET_COLUMNS)
        w.writerows(todas)
    return len(todas)


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF de entrada")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV de saída")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log em nível DEBUG")
    return parser.parse_args()


def main() -> None:
    args = _cli()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    n = processar_pdf(args.pdf, args.out)
    LOG.info("Concluído: %d linhas escritas em %s", n, args.out)


if __name__ == "__main__":
    main()
