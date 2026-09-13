"""Carga e uso do modelo no formato ONNX Runtime.

O contrato de retorno e identico ao de `app.modelo` (sklearn): a camada de
chamada em `main.py` escolhe qual backend usar pela variavel TC3_BACKEND e
chama `carregar` / `classificar` sem precisar saber qual modulo esta por baixo.

Por que ONNX e mais rapido que o sklearn puro?
- O ONNX Runtime compila o grafo de operacoes para kernels nativos otimizados.
- O TF-IDF em ONNX evita alocacoes Python no caminho quente: o resultado e uma
  array densa ou esparsa ja no espaco C++/numpy.
- A regressao logistica em ONNX usa BLAS diretamente, sem o overhead do wrapper
  Python que o sklearn adiciona em cada chamada de predict_proba.
"""

from __future__ import annotations

import time

import numpy as np

from treino import dados, treinar

# Importado aqui e nao no topo do modulo de forma condicional: onnxruntime e
# uma dependencia opcional (nao exigida pela API sklearn), e o import falhar
# em ambiente sem o pacote deve dar ImportError claro, nao AttributeError.
import onnxruntime as rt  # noqa: E402

ARQUIVO_ONNX = "modelo.onnx"


def carregar() -> rt.InferenceSession:
    """Le o artefato ONNX do disco. Chamado uma vez, na subida do processo."""
    caminho = treinar.diretorio_modelo() / ARQUIVO_ONNX
    if not caminho.exists():
        raise OSError(f"artefato ONNX nao encontrado em {caminho}. Execute treino/exportar_onnx.py primeiro.")
    # IntraOpNumThreads=1: o modelo e pequeno e uma requisicao por vez; threads
    # adicionais so adicionam overhead de sincronizacao na escala de microssegundos.
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return rt.InferenceSession(str(caminho), sess_options=opts, providers=["CPUExecutionProvider"])


def classificar(sessao: rt.InferenceSession, texto: str) -> tuple[int, str, float, float]:
    """Devolve (classe_id, nome, confianca, latencia_ms) para um laudo, via ONNX."""
    inicio = time.perf_counter()

    # O modelo ONNX exportado espera StringTensorType de shape [None, 1] ou [None].
    # reshape(-1, 1) produz shape (1, 1).
    entrada = np.array([texto]).reshape(-1, 1)
    nome_entrada = sessao.get_inputs()[0].name

    saidas = sessao.run(None, {nome_entrada: entrada})
    latencia_ms = (time.perf_counter() - inicio) * 1000

    label_previsto = int(saidas[0][0])

    # Se zipmap=False, saidas[1] e um ndarray de formato (1, num_classes).
    # As colunas correspondem a clf.classes_, que sao [1, 2, 3, 4, 5].
    probs_saida = saidas[1]
    if isinstance(probs_saida, np.ndarray):
        # Mapeia classe 1..5 para indice 0..4
        idx = label_previsto - 1 if 1 <= label_previsto <= 5 else 0
        confianca = float(probs_saida[0][idx])
    elif isinstance(probs_saida, list) and len(probs_saida) > 0:
        confianca = float(probs_saida[0].get(label_previsto, 0.0))
    else:
        confianca = 0.0

    return label_previsto, dados.CLASSES[label_previsto], confianca, latencia_ms
