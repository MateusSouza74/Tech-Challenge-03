"""Contrato de entrada e saida da API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LaudoEntrada(BaseModel):
    # min_length=1 e o que faz laudo vazio voltar 422 sem uma linha de validacao
    # escrita a mao no handler.
    texto: str = Field(min_length=1, description="Texto do laudo a classificar")


class ClassificacaoSaida(BaseModel):
    classe: str = Field(description="Nome da categoria de condicao prevista")
    classe_id: int = Field(description="Identificador da categoria, de 1 a 5")
    confianca: float = Field(description="Probabilidade da classe prevista, de 0 a 1")
    latencia_ms: float = Field(description="Tempo da inferencia, sem o overhead HTTP")
