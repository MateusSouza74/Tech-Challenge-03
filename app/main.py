"""API de classificacao de laudos medicos."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from app import modelo as servico
from app.schemas import ClassificacaoSaida, LaudoEntrada

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metricas Prometheus
# ---------------------------------------------------------------------------

REQUISICOES_TOTAL = Counter(
    "laudos_requisicoes_total",
    "Total de requisicoes recebidas pelo endpoint /predict",
    ["metodo", "status"],
)

LATENCIA_INFERENCIA = Histogram(
    "laudos_latencia_inferencia_ms",
    "Latencia da inferencia do modelo (sem overhead HTTP), em milissegundos",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
)

LATENCIA_HTTP = Histogram(
    "laudos_latencia_http_ms",
    "Latencia total da requisicao HTTP ao endpoint /predict, em milissegundos",
    buckets=[1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0],
)

ERROS_TOTAL = Counter(
    "laudos_erros_total",
    "Total de respostas de erro (status >= 400)",
    ["status"],
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # O artefato e lido uma vez, na subida. Carregar por requisicao somaria a leitura
    # de um joblib de 2,4 MB a cada chamada, e e disso que o baseline de latencia
    # tem que estar livre para significar alguma coisa.
    backend = os.environ.get("TC3_BACKEND", "sklearn").lower()
    try:
        if backend == "onnx":
            from app import modelo_onnx  # noqa: PLC0415
            app.state.modelo = modelo_onnx.carregar()
            app.state.backend = "onnx"
        else:
            app.state.modelo = servico.carregar()
            app.state.backend = "sklearn"
    except (OSError, ValueError, ImportError):
        # Nao derruba o processo: a falha realista aqui e a imagem subir sem o
        # diretorio `modelo/`, e um /health devolvendo 503 diz isso em um curl,
        # enquanto um container em crash loop obriga a ir ler log de plataforma.
        logger.exception("falha ao carregar o modelo (backend=%s)", backend)
        app.state.modelo = None
        app.state.backend = backend
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Classificador de laudos medicos",
    description="Recebe o texto de um laudo e devolve a categoria de condicao do paciente.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Readiness de verdade: sem modelo em memoria, o servico nao esta pronto."""
    if app.state.modelo is None:
        raise HTTPException(status_code=503, detail="modelo nao carregado")
    return {"status": "ok", "backend": app.state.backend}


@app.get("/metrics")
def metrics() -> Response:
    """Endpoint de scraping para o Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# `def` e nao `async def`: a inferencia e trabalho de CPU, e o Starlette manda funcao
# sincrona para o threadpool. Como `async def`, ela bloquearia o event loop e a
# latencia sob concorrencia desabaria.
@app.post("/predict", response_model=ClassificacaoSaida)
def predict(entrada: LaudoEntrada, request: Request) -> ClassificacaoSaida:
    """Classifica um laudo em uma das 5 categorias de condicao."""
    if app.state.modelo is None:
        REQUISICOES_TOTAL.labels(metodo="POST", status="503").inc()
        ERROS_TOTAL.labels(status="503").inc()
        raise HTTPException(status_code=503, detail="modelo nao carregado")

    if app.state.backend == "onnx":
        from app import modelo_onnx  # noqa: PLC0415
        classe_id, classe, confianca, latencia_ms = modelo_onnx.classificar(app.state.modelo, entrada.texto)
    else:
        classe_id, classe, confianca, latencia_ms = servico.classificar(app.state.modelo, entrada.texto)

    LATENCIA_INFERENCIA.observe(latencia_ms)
    REQUISICOES_TOTAL.labels(metodo="POST", status="200").inc()

    return ClassificacaoSaida(
        classe=classe,
        classe_id=classe_id,
        confianca=round(confianca, 4),
        latencia_ms=round(latencia_ms, 3),
    )
