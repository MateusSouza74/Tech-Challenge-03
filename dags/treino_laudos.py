"""Ingestao e treino do classificador de laudos.

Tres tasks: baixar e validar o corpus, treinar e avaliar, e barrar a publicacao de um
modelo pior que o que ja esta no ar. O treino real leva 9 segundos, entao esta DAG
treina de verdade, nao simula.

O codigo de treino nao e reimplementado aqui: as tasks chamam os mesmos modulos
`treino.dados` e `treino.treinar` que a API e o CI usam. Duplicar a logica faria a DAG
publicar um modelo diferente do que o `python -m treino.treinar` produz.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowFailException

# A DAG vive em `dags/` e os modulos de treino sao irmaos dela, na raiz do repo, que
# nao entra no sys.path do Airflow sozinho. Em container, montar o repo inteiro (ou
# apontar PYTHONPATH para ele) e o que faz este import funcionar.
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from treino import dados, treinar  # noqa: E402

log = logging.getLogger(__name__)

# Piso de regressao. O baseline medido e 0,5904, e 0,55 deixa folga para variacao
# entre execucoes sem deixar passar degradacao real. Chutar a classe majoritaria
# acerta 0,3328, entao um modelo que caia abaixo de 0,55 ainda estaria longe do
# trivial mas ja teria perdido valor demais para substituir o que esta no ar.
PISO_ACURACIA = 0.55


@dag(
    dag_id="treino_classificador_laudos",
    description="Baixa o Medical Abstracts TC Corpus, treina o classificador e valida a acuracia",
    schedule="@weekly",
    start_date=datetime(2026, 9, 1),
    # Sem catchup: retreinar com o corpus de hoje para cada semana que a DAG passou
    # pausada produziria N modelos identicos, um sobrescrevendo o outro.
    catchup=False,
    max_active_runs=1,
    tags=["tech-challenge", "ml", "treino"],
    doc_md=__doc__,
)
def treino_classificador_laudos():
    @task
    def ingerir() -> dict[str, str]:
        """Garante os dois splits em disco e valida colunas, rotulos e volume."""
        caminhos: dict[str, str] = {}
        for split in dados.ARQUIVOS:
            caminho = dados.baixar(split)
            # Le para validar, e o treino le de novo depois. Sao dois processos
            # diferentes, e passar 14 MB de texto por XCom seria muito pior: o XCom
            # e para ponteiro, nao para dado.
            textos, rotulos = dados.carregar(caminho)
            log.info("%s: %d amostras em %s", split, len(textos), caminho)
            for identificador, nome in dados.CLASSES.items():
                log.info("  %s: %d", nome, rotulos.count(identificador))
            caminhos[split] = str(caminho)
        return caminhos

    @task
    def treinar_modelo(caminhos: dict[str, str]) -> dict:
        """Treina, avalia no split de teste e grava o artefato e as metricas."""
        metricas = treinar.treinar(Path(caminhos["treino"]), Path(caminhos["teste"]))
        log.info(
            "acuracia %.4f | f1 macro %.4f | %.1fs de treino",
            metricas["acuracia"],
            metricas["f1_macro"],
            metricas["segundos_treino"],
        )
        # Sem o relatorio por classe: ele tem 5 dicionarios aninhados e polui o XCom
        # sem ninguem a jusante usar. Fica gravado em metricas.json.
        return {chave: metricas[chave] for chave in ("acuracia", "f1_macro", "baseline_classe_majoritaria")}

    @task
    def validar(metricas: dict) -> None:
        """Barra o modelo que nao vale substituir o que esta no ar."""
        if metricas["acuracia"] < PISO_ACURACIA:
            # Falha definitiva, nao retentativa: treinar de novo com o mesmo corpus da
            # o mesmo modelo, entao repetir a task so atrasaria o alerta.
            raise AirflowFailException(
                f"acuracia {metricas['acuracia']:.4f} abaixo do piso de {PISO_ACURACIA}, "
                f"artefato gravado mas nao deve ser publicado"
            )
        log.info(
            "aprovado: acuracia %.4f, piso %.2f, trivial %.4f",
            metricas["acuracia"],
            PISO_ACURACIA,
            metricas["baseline_classe_majoritaria"],
        )

    validar(treinar_modelo(ingerir()))


# Atribuido, e nao chamado solto: o Airflow encontra a DAG varrendo as globais do
# modulo, e o proprio decorador devolve o objeto para isso ("Return dag object such
# that it's accessible in Globals"). Chamada sem atribuicao dependia do auto-registro
# do Airflow 2, que nao existe mais no 3.
dag_treino_laudos = treino_classificador_laudos()
