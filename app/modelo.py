"""Carga e uso do modelo servido.

Importa de `treino` de proposito: o mapa de classes e o caminho do artefato tem que
ter uma definicao so. Duplicar os nomes das 5 classes aqui criaria a chance de a API
responder um rotulo que o treino nunca produziu.
"""

from __future__ import annotations

import time

import joblib
from sklearn.pipeline import Pipeline

from treino import dados, treinar


def carregar() -> Pipeline:
    """Le o artefato do disco. Chamado uma vez, na subida do processo."""
    return joblib.load(treinar.diretorio_modelo() / treinar.ARQUIVO_MODELO)


def classificar(modelo: Pipeline, texto: str) -> tuple[int, str, float, float]:
    """Devolve (classe_id, nome, confianca, latencia_ms) para um laudo."""
    inicio = time.perf_counter()
    probabilidades = modelo.predict_proba([texto])[0]
    latencia_ms = (time.perf_counter() - inicio) * 1000

    # As classes vem do modelo, nao de um sorted() nosso: se o treino for refeito com
    # um subconjunto de rotulos, a ordem de predict_proba acompanha e esta linha nao.
    indice = int(probabilidades.argmax())
    classe_id = int(modelo.classes_[indice])
    return classe_id, dados.CLASSES[classe_id], float(probabilidades[indice]), latencia_ms
