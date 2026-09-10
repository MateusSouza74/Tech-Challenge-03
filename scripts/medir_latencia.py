"""Baseline de latencia da API.

Mede duas coisas separadas de proposito:

1. inferencia pura, chamando o modelo em processo, sem rede nenhuma;
2. a requisicao HTTP completa contra a API de pe.

A separacao e o que torna a comparacao da etapa 4 legivel. Otimizar o modelo mexe no
numero (1); se so o total HTTP fosse medido, um ganho de 1 ms no modelo ficaria
escondido dentro do overhead de rede e pareceria que a otimizacao nao serviu de nada.
"""

from __future__ import annotations

import argparse
import statistics
import time

import httpx2

from app import modelo as servico
from treino import dados


def percentil(amostras: list[float], fracao: float) -> float:
    ordenadas = sorted(amostras)
    indice = min(int(len(ordenadas) * fracao), len(ordenadas) - 1)
    return ordenadas[indice]


def resumir(nome: str, amostras: list[float]) -> str:
    return (
        f"| {nome} | {statistics.mean(amostras):.2f} | {percentil(amostras, 0.50):.2f} "
        f"| {percentil(amostras, 0.95):.2f} | {percentil(amostras, 0.99):.2f} "
        f"| {max(amostras):.2f} |"
    )


def medir_em_processo(textos: list[str], aquecimento: int) -> list[float]:
    modelo = servico.carregar()
    # A primeira inferencia paga inicializacao preguicosa do scipy e do numpy, e sai
    # uma ordem de grandeza acima das seguintes. Medir sem descartar isso reporta um
    # p99 que nao acontece em regime.
    for texto in textos[:aquecimento]:
        servico.classificar(modelo, texto)

    amostras = []
    for texto in textos:
        inicio = time.perf_counter()
        servico.classificar(modelo, texto)
        amostras.append((time.perf_counter() - inicio) * 1000)
    return amostras


def medir_http(url: str, textos: list[str], aquecimento: int) -> list[float]:
    # Client reaproveitado, e nao uma conexao por requisicao: cliente real mantem
    # keep-alive, e abrir socket a cada chamada mediria o TCP, nao a API.
    with httpx2.Client(base_url=url, timeout=30) as cliente:
        for texto in textos[:aquecimento]:
            cliente.post("/predict", json={"texto": texto})

        amostras = []
        for texto in textos:
            inicio = time.perf_counter()
            resposta = cliente.post("/predict", json={"texto": texto})
            amostras.append((time.perf_counter() - inicio) * 1000)
            resposta.raise_for_status()
    return amostras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="base da API de pe")
    parser.add_argument("--requisicoes", type=int, default=200)
    parser.add_argument("--aquecimento", type=int, default=20)
    argumentos = parser.parse_args()

    # Laudos reais do split de teste, nao texto sintetico: o custo do TF-IDF depende do
    # tamanho do documento, e abstract de verdade tem uns 1.200 caracteres.
    textos, _ = dados.carregar_split("teste")
    textos = textos[: argumentos.requisicoes]

    em_processo = medir_em_processo(textos, argumentos.aquecimento)
    http = medir_http(argumentos.url, textos, argumentos.aquecimento)

    print(f"{len(textos)} laudos, {argumentos.aquecimento} de aquecimento descartados\n")
    print("| Medicao | media | p50 | p95 | p99 | max |")
    print("|---|---:|---:|---:|---:|---:|")
    print(resumir("Inferencia em processo (ms)", em_processo))
    print(resumir("Requisicao HTTP completa (ms)", http))
    print(f"\nOverhead HTTP no p50: {percentil(http, 0.50) - percentil(em_processo, 0.50):.2f} ms")
    print(f"Throughput sequencial: {1000 / statistics.mean(http):.0f} req/s")


if __name__ == "__main__":
    main()
