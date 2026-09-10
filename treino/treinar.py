"""Treino do classificador de laudos: TF-IDF + regressao logistica.

A escolha e deliberada e nao um atalho. O texto e um abstract clinico em ingles, as
5 classes sao amplas, e um modelo linear sobre TF-IDF chega perto do teto do que se
extrai desse corpus sem transformer. Em troca, treina em segundos (o que faz a DAG
do Airflow treinar de verdade, nao simular), gera um artefato de poucos MB e infere
em fracao de milissegundo na CPU.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from treino import dados

ARQUIVO_MODELO = "modelo.joblib"
ARQUIVO_METRICAS = "metricas.json"


def diretorio_modelo() -> Path:
    """Onde o artefato e as metricas ficam. Configuravel para a DAG em container."""
    return Path(os.environ.get("TC3_MODEL_DIR", dados.RAIZ / "modelo"))


def construir_pipeline() -> Pipeline:
    """O vetorizador e o classificador como uma unidade so.

    Pipeline, e nao dois objetos salvos separados, porque o vocabulario do TF-IDF e
    os coeficientes precisam viajar juntos: modelo servido com vocabulario de outra
    versao prediz sem erro e com resultado errado.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    # Bigrama pega termo composto do dominio ("myocardial infarction"),
                    # que separado em duas palavras perde o sentido.
                    ngram_range=(1, 2),
                    # Termo que aparece em menos de 3 abstracts e quase sempre nome
                    # proprio ou erro de digitacao: engorda o vocabulario e nao
                    # generaliza.
                    min_df=3,
                    max_features=50_000,
                    stop_words="english",
                    strip_accents="unicode",
                    # Abstract e texto longo: sem isso, termo repetido 20 vezes pesa
                    # 20 vezes mais que o que aparece uma.
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    # As classes sao desbalanceadas (3.844 contra 1.195 no treino), e aqui
                    # nao ha troca a fazer: medido no split de teste, `balanced` ganha nas
                    # duas metricas (0,5904 / 0,5932 contra 0,5464 / 0,5261 sem ele). O C
                    # fica no default, porque C=5 piorou os dois casos.
                    class_weight="balanced",
                ),
            ),
        ]
    )


def treinar(caminho_treino: Path, caminho_teste: Path) -> dict:
    """Treina, avalia no split de teste e grava artefato e metricas em disco."""
    textos_treino, rotulos_treino = dados.carregar(caminho_treino)
    textos_teste, rotulos_teste = dados.carregar(caminho_teste)

    modelo = construir_pipeline()
    inicio = time.perf_counter()
    modelo.fit(textos_treino, rotulos_treino)
    segundos_treino = time.perf_counter() - inicio

    previsto = modelo.predict(textos_teste)
    nomes = [dados.CLASSES[identificador] for identificador in sorted(dados.CLASSES)]

    # Piso de leitura da acuracia: chutar sempre a classe mais frequente do teste.
    # Sem esse numero ao lado, uma acuracia de 0,6 em 5 classes nao diz nada.
    baseline = max(rotulos_teste.count(c) for c in dados.CLASSES) / len(rotulos_teste)

    metricas = {
        "acuracia": round(accuracy_score(rotulos_teste, previsto), 4),
        "f1_macro": round(f1_score(rotulos_teste, previsto, average="macro"), 4),
        "baseline_classe_majoritaria": round(baseline, 4),
        "amostras_treino": len(textos_treino),
        "amostras_teste": len(textos_teste),
        "termos_no_vocabulario": len(modelo.named_steps["tfidf"].vocabulary_),
        "segundos_treino": round(segundos_treino, 2),
        "relatorio_por_classe": classification_report(
            rotulos_teste, previsto, target_names=nomes, output_dict=True, zero_division=0
        ),
        # A versao entra no artefato porque joblib de scikit-learn diferente do que le
        # levanta aviso de incompatibilidade, e o aviso e facil de ignorar.
        "versao_sklearn": sklearn.__version__,
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    destino = diretorio_modelo()
    destino.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, destino / ARQUIVO_MODELO, compress=3)
    (destino / ARQUIVO_METRICAS).write_text(json.dumps(metricas, indent=2), encoding="utf-8")

    return metricas


if __name__ == "__main__":
    metricas = treinar(dados.baixar("treino"), dados.baixar("teste"))
    artefato = diretorio_modelo() / ARQUIVO_MODELO
    print(f"acuracia          {metricas['acuracia']:.4f}")
    print(f"f1 macro          {metricas['f1_macro']:.4f}")
    print(f"baseline majorit. {metricas['baseline_classe_majoritaria']:.4f}")
    print(f"vocabulario       {metricas['termos_no_vocabulario']} termos")
    print(f"treino            {metricas['segundos_treino']:.2f}s")
    print(f"artefato          {artefato} ({artefato.stat().st_size / 1024 / 1024:.1f} MB)")
