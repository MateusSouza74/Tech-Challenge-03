"""Testes da carga de dados e do pipeline de treino.

O smoke de treino roda sobre a amostra de 200 linhas versionada em `dados/`, nao sobre
o corpus completo: leva cerca de um segundo e nao precisa de rede, e o que ele prova e
que o pipeline monta, ajusta e preve, nao que o modelo e bom.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from treino import dados, treinar

AMOSTRA = dados.RAIZ / "dados" / "amostra_laudos.csv"


def escrever_csv(caminho: Path, cabecalho: list[str], linhas: list[tuple]) -> Path:
    with open(caminho, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(cabecalho)
        escritor.writerows(linhas)
    return caminho


def test_mapa_de_classes_cobre_os_cinco_rotulos():
    assert sorted(dados.CLASSES) == [1, 2, 3, 4, 5]


def test_amostra_versionada_carrega_balanceada():
    textos, rotulos = dados.carregar(AMOSTRA, minimo=200)
    assert len(textos) == 200
    assert {rotulo: rotulos.count(rotulo) for rotulo in sorted(dados.CLASSES)} == dict.fromkeys(dados.CLASSES, 40)


def test_carregar_recusa_csv_sem_as_colunas_esperadas(tmp_path):
    caminho = escrever_csv(tmp_path / "errado.csv", ["label", "text"], [(1, "abc")])
    with pytest.raises(ValueError, match="nao tem as colunas"):
        dados.carregar(caminho, minimo=1)


def test_carregar_recusa_rotulo_fora_das_classes(tmp_path):
    caminho = escrever_csv(tmp_path / "rotulo.csv", [dados.COLUNA_ROTULO, dados.COLUNA_TEXTO], [(9, "abc")])
    with pytest.raises(ValueError, match="fora de"):
        dados.carregar(caminho, minimo=1)


def test_carregar_recusa_arquivo_abaixo_do_minimo(tmp_path):
    # E a checagem que faz a ingestao da DAG falhar alto se a fonte mudar ou o
    # download vier truncado, em vez de treinar um modelo em 3 linhas.
    caminho = escrever_csv(tmp_path / "curto.csv", [dados.COLUNA_ROTULO, dados.COLUNA_TEXTO], [(1, "abc")])
    with pytest.raises(ValueError, match="abaixo do minimo"):
        dados.carregar(caminho, minimo=2000)


def test_pipeline_treina_e_preve_classe_valida():
    textos, rotulos = dados.carregar(AMOSTRA, minimo=200)
    modelo = treinar.construir_pipeline()
    modelo.fit(textos, rotulos)

    previsto = modelo.predict(textos[:5])
    assert all(int(rotulo) in dados.CLASSES for rotulo in previsto)
