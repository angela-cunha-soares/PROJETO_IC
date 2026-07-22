"""Runner mestre - executa todas as etapas do cronograma na ordem correta.

Encadeia os pipelines de cada etapa (2 a 7 do cronograma) como subprocessos
isolados, respeitando as dependencias entre eles:

    2. Pre-processamento .......... run_preprocess.py
    3. Outliers / anomalias ....... run_outliers.py
    4. Analise descritiva (EDA) ... run_descritiva.py
    5. Aplicacao de modelos ....... run_train.py
    6. Metricas / avaliacao ....... run_evaluate.py
    7. Relatorio final (Word) ..... gerar_relatorio_docx.py

A etapa 1 (revisao bibliografica) e textual e nao possui script.

Uso:
    python scripts/run_all.py            # roda tudo
    python scripts/run_all.py --list     # lista as etapas e sai
    python scripts/run_all.py --only 5 6 # roda apenas essas etapas
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
LOG = logging.getLogger("run_all")

# (rotulo, arquivo) na ordem de dependencia.
ETAPAS: list[tuple[str, str]] = [
    ("Pre-processamento e imputacao", "run_preprocess.py"),
    ("Validacao da imputacao (mascaramento)", "validar_imputacao.py"),
    ("Deteccao de outliers/anomalias", "run_outliers.py"),
    ("Analise descritiva (EDA)", "run_descritiva.py"),
    ("Aplicacao de modelos (K-Means, RF, IForest)", "run_train.py"),
    ("Metricas e figuras de avaliacao", "run_evaluate.py"),
    ("Relatorio final (Word .docx)", "gerar_relatorio_docx.py"),
]


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", type=int, nargs="+", metavar="N",
                   help="executa apenas as etapas indicadas (1..6 desta lista)")
    p.add_argument("--list", action="store_true", help="lista as etapas e sai")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _cli()

    if args.list:
        for i, (rotulo, arq) in enumerate(ETAPAS, 1):
            print(f"{i}. {rotulo}  ({arq})")
        return 0

    indices = args.only or list(range(1, len(ETAPAS) + 1))
    falhas: list[str] = []

    for i in indices:
        if not 1 <= i <= len(ETAPAS):
            LOG.warning("Etapa %s inexistente - ignorada.", i)
            continue
        rotulo, arq = ETAPAS[i - 1]
        LOG.info("========== [%d/%d] %s (%s) ==========", i, len(ETAPAS), rotulo, arq)
        res = subprocess.run([sys.executable, str(SCRIPTS / arq)], cwd=PROJECT_ROOT)
        if res.returncode != 0:
            LOG.error("Falhou: %s (codigo %d)", arq, res.returncode)
            falhas.append(arq)

    if falhas:
        LOG.error("Concluido com falhas em: %s", ", ".join(falhas))
        return 1
    LOG.info("Pipeline completo executado com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
