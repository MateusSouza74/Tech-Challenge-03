"""API de classificacao de laudos medicos."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import modelo as servico
from app.schemas import ClassificacaoSaida, LaudoEntrada

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # O artefato e lido uma vez, na subida. Carregar por requisicao somaria a leitura
    # de um joblib de 2,4 MB a cada chamada, e e disso que o baseline de latencia
    # tem que estar livre para significar alguma coisa.
    try:
        app.state.modelo = servico.carregar()
    except (OSError, ValueError):
        # Nao derruba o processo: a falha realista aqui e a imagem subir sem o
        # diretorio `modelo/`, e um /health devolvendo 503 diz isso em um curl,
        # enquanto um container em crash loop obriga a ir ler log de plataforma.
        logger.exception("falha ao carregar o modelo")
        app.state.modelo = None
    yield


app = FastAPI(
    title="Classificador de laudos medicos",
    description="Recebe o texto de um laudo e devolve a categoria de condicao do paciente.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Readiness de verdade: sem modelo em memoria, o servico nao esta pronto."""
    if app.state.modelo is None:
        raise HTTPException(status_code=503, detail="modelo nao carregado")
    return {"status": "ok"}


# `def` e nao `async def`: a inferencia e trabalho de CPU, e o Starlette manda funcao
# sincrona para o threadpool. Como `async def`, ela bloquearia o event loop e a
# latencia sob concorrencia desabaria.
@app.post("/predict", response_model=ClassificacaoSaida)
def predict(entrada: LaudoEntrada) -> ClassificacaoSaida:
    """Classifica um laudo em uma das 5 categorias de condicao."""
    if app.state.modelo is None:
        raise HTTPException(status_code=503, detail="modelo nao carregado")

    classe_id, classe, confianca, latencia_ms = servico.classificar(app.state.modelo, entrada.texto)
    return ClassificacaoSaida(
        classe=classe,
        classe_id=classe_id,
        confianca=round(confianca, 4),
        latencia_ms=round(latencia_ms, 3),
    )
