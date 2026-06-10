"""Pipeline completo de análise de outliers — roda todos os scripts em ordem.

Encadeia (cada um como subprocesso isolado, na ordem de dependência):

1. ``comparar_metodos_outliers.py``   — métricas de comparação interna + figuras ``metodos_*``.
2. ``melhor_metodo_por_variavel.py``  — melhor método por variável (CSV + figuras).
3. ``plot_outliers.py``               — classificação em 2 camadas (figuras ``outliers_*``).
4. ``tabela_outliers_por_metodo.py``  — tabelas de valores + ``outliers_confirmados.csv``.
5. ``plot_outliers_por_metodo.py``    — painel PCA-2D por método + CONAMA.
6. ``plot_outliers_confirmados.py``   — heatmap/timeline/com-sem-Fe dos confirmados.

O passo 6 depende do 4 (lê ``outliers_confirmados.csv``); por isso a ordem é fixa.

Uso:
    python scripts/run_outliers.py              # roda todos
    python scripts/run_outliers.py --only 4 5 6 # roda só esses passos
    python scripts/run_outliers.py --list       # lista os passos e sai
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
LOG = logging.getLogger("run_outliers")

# (rótulo, arquivo) na ordem de execução.
PASSOS: list[tuple[str, str]] = [
    ("Comparação interna de métodos", "comparar_metodos_outliers.py"),
    ("Melhor método por variável", "melhor_metodo_por_variavel.py"),
    ("Classificação em 2 camadas", "plot_outliers.py"),
    ("Tabelas de valores + confirmados", "tabela_outliers_por_metodo.py"),
    ("Painel PCA-2D por método", "plot_outliers_por_metodo.py"),
    ("Figuras dos confirmados", "plot_outliers_confirmados.py"),
    ("Calibração de contamination (CV)", "calibrar_contamination.py"),
]


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", type=int, nargs="+", metavar="N",
                   help="executa só os passos indicados (1..6)")
    p.add_argument("--list", action="store_true", help="lista os passos e sai")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _cli()

    if args.list:
        for i, (rotulo, arq) in enumerate(PASSOS, 1):
            print(f"{i}. {rotulo}  ({arq})")
        return 0

    indices = args.only or list(range(1, len(PASSOS) + 1))
    falhas: list[str] = []

    for i in indices:
        if not 1 <= i <= len(PASSOS):
            LOG.warning("Passo %s inexistente — ignorado.", i)
            continue
        rotulo, arq = PASSOS[i - 1]
        LOG.info("=== [%d/%d] %s (%s) ===", i, len(PASSOS), rotulo, arq)
        res = subprocess.run([sys.executable, str(SCRIPTS / arq)], cwd=PROJECT_ROOT)
        if res.returncode != 0:
            LOG.error("Falhou: %s (código %d)", arq, res.returncode)
            falhas.append(arq)

    if falhas:
        LOG.error("Concluído com falhas em: %s", ", ".join(falhas))
        return 1
    LOG.info("Análise de outliers concluída com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
