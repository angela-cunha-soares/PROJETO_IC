"""Converte o relatório de resultados (Markdown) em PDF, com texto, tabelas e figuras.

Lê ``reports/relatorio_resultados.md`` e renderiza um PDF paginado A4 usando
matplotlib (``PdfPages``) — sem dependência externa de conversão. Suporta:
títulos (#/##/###), parágrafos, listas, citações (>), tabelas em pipe (| … |)
e imagens (``![alt](figures/x.png)``), com legendas em itálico.

Saída: ``reports/relatorio_resultados.pdf``.

Uso:
    python scripts/relatorio_md_para_pdf.py
    python scripts/relatorio_md_para_pdf.py --md reports/relatorio_resultados.md
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config  # noqa: E402

LOG = logging.getLogger("relatorio_md_para_pdf")

# Geometria da página (frações da figura A4 8.27 × 11.69 in)
PAGE_W, PAGE_H = 8.27, 11.69
LEFT, TOP, BOTTOM = 0.08, 0.95, 0.06
WIDTH = 0.86
WRAP = 100          # largura de quebra de texto (caracteres)
LINE_H = 0.0155     # altura de linha do corpo


def _clean(text: str) -> str:
    """Remove marcações inline do Markdown, preservando o texto legível."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)            # imagens soltas
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links → texto
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)  # *itálico*
    text = text.replace("`", "")
    return text.strip()


# --- Parsing do Markdown em blocos --------------------------------------------

def parse_blocos(md: str) -> list[tuple]:
    """Converte o Markdown numa lista de blocos ``(tipo, dados)``."""
    linhas = md.splitlines()
    blocos: list[tuple] = []
    i, n = 0, len(linhas)
    while i < n:
        ln = linhas[i]
        stripped = ln.strip()

        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):  # ignora blocos de código cercados
            i += 1
            while i < n and not linhas[i].strip().startswith("```"):
                i += 1
            i += 1
            continue
        if set(stripped) <= {"-"} and len(stripped) >= 3:  # regra horizontal ---
            blocos.append(("hr", None))
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            blocos.append(("h", (len(m.group(1)), _clean(m.group(2)))))
            i += 1
            continue
        if stripped.startswith("!["):  # imagem
            mm = re.match(r"^!\[(.*?)\]\((.*?)\)", stripped)
            src = mm.group(2) if mm else ""
            legenda = ""
            if i + 1 < n and linhas[i + 1].strip().startswith(("**Figura", "**Tabela")):
                legenda = _clean(linhas[i + 1].strip())
                i += 1
            blocos.append(("img", (src, legenda)))
            i += 1
            continue
        if stripped.startswith("|"):  # tabela pipe
            tbl = []
            while i < n and linhas[i].strip().startswith("|"):
                tbl.append(linhas[i].strip())
                i += 1
            blocos.append(("table", _parse_tabela(tbl)))
            continue
        if stripped.startswith(">"):  # citação (pode ter várias linhas)
            quote = []
            while i < n and linhas[i].strip().startswith(">"):
                quote.append(linhas[i].strip().lstrip(">").strip())
                i += 1
            blocos.append(("quote", _clean(" ".join(quote))))
            continue
        if re.match(r"^[-*●]\s+", stripped):  # item de lista
            blocos.append(("li", _clean(re.sub(r"^[-*●]\s+", "", stripped))))
            i += 1
            continue
        # parágrafo: agrega linhas consecutivas "normais"
        par = [stripped]
        i += 1
        while i < n and linhas[i].strip() and not re.match(
            r"^(#|\||>|!\[|[-*●]\s|```)", linhas[i].strip()
        ) and not (set(linhas[i].strip()) <= {"-"} and len(linhas[i].strip()) >= 3):
            par.append(linhas[i].strip())
            i += 1
        blocos.append(("p", _clean(" ".join(par))))
    return blocos


def _parse_tabela(linhas: list[str]) -> tuple[list[str], list[list[str]]]:
    def cells(row: str) -> list[str]:
        return [_clean(c) for c in row.strip().strip("|").split("|")]
    header = cells(linhas[0])
    rows = [cells(r) for r in linhas[2:]]  # pula a linha separadora |---|
    return header, rows


# --- Renderizador paginado ----------------------------------------------------

class Renderer:
    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self._new_page()

    def _new_page(self) -> None:
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.y = TOP

    def _flush(self) -> None:
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def _ensure(self, h: float) -> None:
        if self.y - h < BOTTOM:
            self._flush()
            self._new_page()

    def heading(self, level: int, text: str) -> None:
        size = {1: 16, 2: 13.5, 3: 11.5}.get(level, 10.5)
        gap = 0.022 if level <= 2 else 0.016
        self.y -= gap
        self._ensure(0.03)
        self.fig.text(LEFT, self.y, text, fontsize=size, fontweight="bold", va="top")
        self.y -= 0.026 if level <= 2 else 0.020

    def paragraph(self, text: str, *, italic=False, indent=0.0, size=9.3) -> None:
        cap = text.startswith(("Figura", "Tabela"))
        ital = italic or cap
        for ln in textwrap.wrap(text, WRAP) or [""]:
            self._ensure(LINE_H)
            self.fig.text(LEFT + indent, self.y, ln, fontsize=size, va="top",
                          style="italic" if ital else "normal",
                          color="#333" if cap else "black")
            self.y -= LINE_H
        self.y -= 0.005

    def bullet(self, text: str) -> None:
        linhas = textwrap.wrap(text, WRAP - 3) or [""]
        for k, ln in enumerate(linhas):
            self._ensure(LINE_H)
            prefix = "•  " if k == 0 else "   "
            self.fig.text(LEFT + 0.01, self.y, prefix + ln, fontsize=9.3, va="top")
            self.y -= LINE_H
        self.y -= 0.003

    def quote(self, text: str) -> None:
        for ln in textwrap.wrap(text, WRAP - 4) or [""]:
            self._ensure(LINE_H)
            self.fig.text(LEFT + 0.02, self.y, ln, fontsize=8.8, va="top",
                          style="italic", color="#444")
            self.y -= LINE_H
        self.y -= 0.006

    def image(self, src: str, legenda: str, base: Path) -> None:
        path = (base / src).resolve()
        if not path.is_file():
            LOG.warning("Imagem não encontrada: %s", path)
            return
        img = mpimg.imread(path)
        aspect = img.shape[0] / img.shape[1]
        disp_w_in = WIDTH * PAGE_W
        disp_h_in = min(disp_w_in * aspect, 4.3)      # limita altura
        disp_w_in = disp_h_in / aspect                # mantém proporção
        w_frac, h_frac = disp_w_in / PAGE_W, disp_h_in / PAGE_H
        self._ensure(h_frac + 0.01)
        x = LEFT + (WIDTH - w_frac) / 2               # centraliza
        ax = self.fig.add_axes([x, self.y - h_frac, w_frac, h_frac])
        ax.imshow(img)
        ax.set_aspect("auto")
        ax.axis("off")
        self.y -= h_frac + 0.006
        if legenda:
            self.paragraph(legenda, italic=True, size=8.5)

    def table(self, header: list[str], rows: list[list[str]]) -> None:
        nrows = len(rows) + 1
        row_h = 0.020
        h_tab = nrows * row_h
        if h_tab > (TOP - BOTTOM):     # tabela maior que a página: quebra em partes
            meio = len(rows) // 2
            self.table(header, rows[:meio])
            self.table(header, rows[meio:])
            return
        self._ensure(h_tab + 0.012)
        ax = self.fig.add_axes([LEFT, self.y - h_tab, WIDTH, h_tab])
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.scale(1, 1.15)
        for (r, _c), cell in t.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#eaeaea")
        self.y -= h_tab + 0.012

    def render(self, blocos: list[tuple], base: Path) -> None:
        for tipo, dados in blocos:
            if tipo == "h":
                self.heading(*dados)
            elif tipo == "p":
                self.paragraph(dados)
            elif tipo == "li":
                self.bullet(dados)
            elif tipo == "quote":
                self.quote(dados)
            elif tipo == "img":
                self.image(dados[0], dados[1], base)
            elif tipo == "table":
                self.table(*dados)
            elif tipo == "hr":
                self.y -= 0.006
        self._flush()


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--md", type=Path,
                   default=config.REPORTS_DIR / "relatorio_resultados.md")
    p.add_argument("--out", type=Path,
                   default=config.REPORTS_DIR / "relatorio_resultados.pdf")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _cli()
    if not args.md.is_file():
        raise FileNotFoundError(f"Markdown não encontrado: {args.md}")
    blocos = parse_blocos(args.md.read_text(encoding="utf-8"))
    with PdfPages(args.out) as pdf:
        Renderer(pdf).render(blocos, base=args.md.parent)
    LOG.info("PDF gerado: %s (%d blocos)", args.out, len(blocos))


if __name__ == "__main__":
    main()
