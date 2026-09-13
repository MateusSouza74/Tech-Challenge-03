"""Configuracao global do pytest.

Garante que a raiz do projeto (onde estao os pacotes 'app' e 'treino') esteja no sys.path
independentemente de onde o comando `pytest` ou `python -m pytest` for executado.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
