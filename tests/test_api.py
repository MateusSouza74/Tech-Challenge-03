"""Testes da API.

Sem rede e sem Docker de proposito: e isso que permite rodarem no CI. O modelo lido
e o artefato versionado em `modelo/`, e a amostra de laudos vem do CSV de 200 linhas
que tambem esta no repo.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from treino import dados

LAUDO = (
    "Acute myocardial infarction in a patient with severe coronary artery stenosis. "
    "Cardiac catheterization revealed occlusion of the left anterior descending artery."
)


@pytest.fixture
def cliente():
    # Escopo de funcao, nao de modulo: o teste do caminho degradado sobe um cliente
    # proprio que zera `app.state.modelo`, e como o `app` e um objeto unico do modulo,
    # um cliente compartilhado herdaria esse estado dependendo da ordem dos testes.
    with TestClient(app) as instancia:
        yield instancia


def test_health_responde_ok(cliente):
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_predict_devolve_classe_conhecida(cliente):
    corpo = cliente.post("/predict", json={"texto": LAUDO}).json()
    assert corpo["classe"] in dados.CLASSES.values()
    assert corpo["classe_id"] in dados.CLASSES
    # O nome e o id tem que se referir a mesma classe, senao a resposta e incoerente
    # mesmo com os dois campos individualmente validos.
    assert dados.CLASSES[corpo["classe_id"]] == corpo["classe"]


def test_predict_devolve_confianca_e_latencia_plausiveis(cliente):
    corpo = cliente.post("/predict", json={"texto": LAUDO}).json()
    assert 0.0 <= corpo["confianca"] <= 1.0
    assert corpo["latencia_ms"] > 0


def test_predict_recusa_texto_vazio(cliente):
    assert cliente.post("/predict", json={"texto": ""}).status_code == 422


def test_predict_recusa_corpo_sem_o_campo(cliente):
    assert cliente.post("/predict", json={}).status_code == 422


def test_sem_artefato_em_disco_os_dois_endpoints_dao_503(monkeypatch, tmp_path):
    # A falha realista que isto cobre e a imagem subir sem o diretorio `modelo/`, por
    # um .dockerignore mal editado. O contrato e responder 503, nao entrar em crash
    # loop nem responder 200 sem modelo.
    monkeypatch.setenv("TC3_MODEL_DIR", str(tmp_path))
    with TestClient(app) as sem_modelo:
        assert sem_modelo.get("/health").status_code == 503
        assert sem_modelo.post("/predict", json={"texto": LAUDO}).status_code == 503
