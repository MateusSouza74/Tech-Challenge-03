"""Exporta o pipeline treinado (sklearn) para o formato ONNX.

Por que ONNX?
- Separacao de treinamento e inferencia: o ONNX Runtime carrega o grafo compilado
  sem depender do scikit-learn em producao (mas aqui mantemos o sklearn na imagem
  porque o TF-IDF ainda e usado no treino e na DAG).
- Kernels nativos otimizados: o Runtime compila operacoes de algebra linear para
  BLAS direto, eliminando o overhead do wrapper Python do sklearn.
- Portabilidade: o .onnx pode ser servido por qualquer runtime compativel (Triton,
  TorchServe, Azure ML, etc.) sem reescrita.

Uso:
    python -m treino.exportar_onnx
    # Grava modelo/modelo.onnx ao lado do modelo.joblib existente.

    python -m treino.exportar_onnx --entrada /outro/modelo.joblib --saida /outro/modelo.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from treino import treinar


def exportar(caminho_joblib: Path, caminho_onnx: Path) -> None:
    """Converte o pipeline sklearn salvo em disco para ONNX e grava em caminho_onnx."""
    pipeline = joblib.load(caminho_joblib)

    # O TF-IDF do pipeline espera uma lista de strings. O tipo de entrada para o
    # conversor e StringTensorType com shape [None, 1]: None = batch dinamico,
    # 1 = uma string por amostra (o TF-IDF achata internamente para 1-D).
    tipo_entrada = [("texto", StringTensorType([None, 1]))]

    modelo_onnx = convert_sklearn(
        pipeline,
        initial_types=tipo_entrada,
        # target_opset=17 e compativel com onnxruntime >= 1.15, que e o que usamos.
        target_opset=17,
        # Opcao que preserva as probabilidades de todas as classes na saida,
        # necessario para devolver `confianca` correto em modelo_onnx.classificar.
        options={type(pipeline.named_steps["clf"]): {"zipmap": False}},
    )

    caminho_onnx.parent.mkdir(parents=True, exist_ok=True)
    caminho_onnx.write_bytes(modelo_onnx.SerializeToString())
    tamanho_kb = caminho_onnx.stat().st_size / 1024
    print(f"exportado: {caminho_onnx}  ({tamanho_kb:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    dir_modelo = treinar.diretorio_modelo()
    parser.add_argument(
        "--entrada",
        type=Path,
        default=dir_modelo / treinar.ARQUIVO_MODELO,
        help="caminho do .joblib a converter (default: modelo/modelo.joblib)",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=dir_modelo / "modelo.onnx",
        help="caminho de saida do .onnx (default: modelo/modelo.onnx)",
    )
    args = parser.parse_args()

    if not args.entrada.exists():
        raise SystemExit(f"artefato nao encontrado: {args.entrada}\nExecute primeiro: python -m treino.treinar")

    exportar(args.entrada, args.saida)


if __name__ == "__main__":
    main()
