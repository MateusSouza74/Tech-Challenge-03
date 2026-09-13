"""Baseline de latencia da API e comparacao sklearn vs ONNX.

Mede tres coisas separadas de proposito:

1. inferencia pura sklearn, chamando o modelo joblib em processo, sem rede;
2. inferencia pura ONNX Runtime, chamando o session.run em processo, sem rede;
3. a requisicao HTTP completa contra a API de pe.

A separacao e o que torna a comparacao da etapa 4 legivel. Otimizar o modelo mexe
nos numeros (1) e (2); se so o total HTTP fosse medido, um ganho de 0,5 ms no
modelo ficaria escondido dentro do overhead de rede e pareceria que a otimizacao
nao serviu de nada.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import httpx  # noqa: E402

from app import modelo as servico  # noqa: E402
from treino import dados  # noqa: E402


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


def medir_sklearn(textos: list[str], aquecimento: int) -> list[float]:
    """Mede a latencia de inferencia do modelo sklearn (joblib) em processo."""
    modelo = servico.carregar()
    for texto in textos[:aquecimento]:
        servico.classificar(modelo, texto)

    amostras = []
    for texto in textos:
        inicio = time.perf_counter()
        servico.classificar(modelo, texto)
        amostras.append((time.perf_counter() - inicio) * 1000)
    return amostras


def medir_onnx(textos: list[str], aquecimento: int) -> list[float] | None:
    """Mede a latencia de inferencia do modelo ONNX Runtime em processo.

    Retorna None se o artefato ONNX nao existir ou o onnxruntime nao estiver instalado.
    """
    try:
        from app import modelo_onnx  # noqa: PLC0415
        sessao = modelo_onnx.carregar()
    except (ImportError, OSError) as exc:
        print(f"[aviso] ONNX nao disponivel: {exc}")
        return None

    for texto in textos[:aquecimento]:
        modelo_onnx.classificar(sessao, texto)

    amostras = []
    for texto in textos:
        inicio = time.perf_counter()
        modelo_onnx.classificar(sessao, texto)
        amostras.append((time.perf_counter() - inicio) * 1000)
    return amostras


def medir_http(url: str, textos: list[str], aquecimento: int) -> list[float]:
    # Client reaproveitado, e nao uma conexao por requisicao: cliente real mantem
    # keep-alive, e abrir socket a cada chamada mediria o TCP, nao a API.
    with httpx.Client(base_url=url, timeout=30) as cliente:
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
    parser.add_argument("--sem-http", action="store_true", help="pula a medicao HTTP (API nao precisa estar de pe)")
    argumentos = parser.parse_args()

    # Laudos reais do split de teste, nao texto sintetico: o custo do TF-IDF depende do
    # tamanho do documento, e abstract de verdade tem uns 1.200 caracteres.
    textos, _ = dados.carregar_split("teste")
    textos = textos[: argumentos.requisicoes]

    print(f"{len(textos)} laudos, {argumentos.aquecimento} de aquecimento descartados\n")
    print("| Medicao | media | p50 | p95 | p99 | max |")
    print("|---|---:|---:|---:|---:|---:|")

    sklearn_amostras = medir_sklearn(textos, argumentos.aquecimento)
    print(resumir("Sklearn — Inferencia em processo (ms)", sklearn_amostras))

    onnx_amostras = medir_onnx(textos, argumentos.aquecimento)
    if onnx_amostras is not None:
        print(resumir("ONNX Runtime — Inferencia em processo (ms)", onnx_amostras))

    if not argumentos.sem_http:
        http_amostras = medir_http(argumentos.url, textos, argumentos.aquecimento)
        print(resumir("Requisicao HTTP completa (ms)", http_amostras))
        print(f"\nOverhead HTTP no p50: {percentil(http_amostras, 0.50) - percentil(sklearn_amostras, 0.50):.2f} ms")
        print(f"Throughput sequencial: {1000 / statistics.mean(http_amostras):.0f} req/s")

    if onnx_amostras is not None:
        p50_sklearn = percentil(sklearn_amostras, 0.50)
        p50_onnx = percentil(onnx_amostras, 0.50)
        ganho = (p50_sklearn - p50_onnx) / p50_sklearn * 100
        print("\n--- Comparacao sklearn vs ONNX ---")
        print(f"p50 sklearn : {p50_sklearn:.2f} ms")
        print(f"p50 ONNX    : {p50_onnx:.2f} ms")
        print(f"Reducao p50 : {ganho:.1f}%")


if __name__ == "__main__":
    main()
