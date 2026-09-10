"""Checagem de estrutura da DAG.

Pega import quebrado, erro de sintaxe, argumento invalido do decorador e task
desconectada do grafo, tudo sem Airflow de pe e sem banco de metadados.

Roda no job proprio do CI, que instala o Airflow com o constraints oficial. Fora dele
o arquivo e pulado: o Airflow nao esta no requirements-dev (inflaria o job de lint e
teste) e nem importa no Windows, porque `ObjectStoragePath` usa `os.register_at_fork`,
que so existe em POSIX.
"""

from __future__ import annotations

import pytest

pytest.importorskip("airflow.sdk", reason="Airflow so e instalado no job de DAG do CI")

import treino_laudos  # noqa: E402

TASKS_ESPERADAS = {"ingerir", "treinar_modelo", "validar"}


@pytest.fixture(scope="module")
def dag():
    return treino_laudos.dag_treino_laudos


def test_dag_esta_nas_globais_do_modulo(dag):
    # Se a chamada do decorador deixar de ser atribuida, o Airflow nao acha a DAG e o
    # sintoma e uma pasta de DAGs silenciosamente vazia, sem erro de import.
    assert dag is not None
    assert dag.dag_id == "treino_classificador_laudos"


def test_dag_tem_as_tres_tasks(dag):
    assert {tarefa.task_id for tarefa in dag.tasks} == TASKS_ESPERADAS


def test_ordem_e_ingerir_treinar_validar(dag):
    assert dag.get_task("ingerir").downstream_task_ids == {"treinar_modelo"}
    assert dag.get_task("treinar_modelo").downstream_task_ids == {"validar"}
    assert dag.get_task("validar").downstream_task_ids == set()


def test_nao_faz_catchup(dag):
    # Retreinar com o corpus de hoje para cada semana pausada geraria N modelos
    # identicos, um sobrescrevendo o outro.
    assert dag.catchup is False
