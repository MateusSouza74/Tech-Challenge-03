"""Carga do Medical Abstracts TC Corpus.

O corpus vive em https://github.com/sebischair/Medical-Abstracts-TC-Corpus, com o
split de treino e teste ja definido pelos autores. Usar o split original evita
discussao sobre vazamento de dados na avaliacao.

Leitura pela stdlib, sem pandas: o CSV tem duas colunas e a unica sutileza sao os
abstracts com virgula e aspas dentro, que o modulo `csv` ja trata.
"""

from __future__ import annotations

import csv
import os
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

URL_BASE = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
ARQUIVOS = {"treino": "medical_tc_train.csv", "teste": "medical_tc_test.csv"}

COLUNA_ROTULO = "condition_label"
COLUNA_TEXTO = "medical_abstract"

CLASSES = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

# Piso exigido pelo enunciado do desafio. O split de treino tem 11.550 linhas e o de
# teste 2.888, entao os dois passam com folga; a checagem existe para a DAG falhar
# alto se a fonte mudar ou o download vier truncado.
MINIMO_AMOSTRAS = 2000


def diretorio_dados() -> Path:
    """Onde os CSVs baixados ficam. Configuravel para a DAG rodar em container."""
    return Path(os.environ.get("TC3_DATA_DIR", RAIZ / "dados"))


def baixar(split: str) -> Path:
    """Baixa o CSV do split se ele ainda nao estiver em disco e devolve o caminho."""
    if split not in ARQUIVOS:
        raise ValueError(f"split desconhecido: {split!r}. Esperado um de {sorted(ARQUIVOS)}")

    destino = diretorio_dados() / ARQUIVOS[split]
    if destino.exists():
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    # Grava num temporario e renomeia: download interrompido nao deixa CSV parcial
    # em disco para o cache servir na proxima execucao.
    temporario = destino.with_suffix(".parcial")
    with urllib.request.urlopen(f"{URL_BASE}/{ARQUIVOS[split]}", timeout=120) as resposta:
        temporario.write_bytes(resposta.read())
    temporario.replace(destino)
    return destino


def carregar(caminho: Path, minimo: int = MINIMO_AMOSTRAS) -> tuple[list[str], list[int]]:
    """Le um CSV do corpus e devolve (textos, rotulos), validando o conteudo."""
    with open(caminho, encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        faltando = {COLUNA_ROTULO, COLUNA_TEXTO} - set(leitor.fieldnames or [])
        if faltando:
            raise ValueError(f"{caminho.name} nao tem as colunas {sorted(faltando)}")

        textos: list[str] = []
        rotulos: list[int] = []
        for numero, linha in enumerate(leitor, start=2):
            rotulo = int(linha[COLUNA_ROTULO])
            if rotulo not in CLASSES:
                raise ValueError(f"{caminho.name}:{numero} tem rotulo {rotulo}, fora de {sorted(CLASSES)}")
            rotulos.append(rotulo)
            textos.append(linha[COLUNA_TEXTO])

    if len(textos) < minimo:
        raise ValueError(f"{caminho.name} tem {len(textos)} amostras, abaixo do minimo de {minimo}")

    return textos, rotulos


def carregar_split(split: str) -> tuple[list[str], list[int]]:
    """Atalho para o caso comum: garante o CSV em disco e ja devolve os dados."""
    return carregar(baixar(split))


if __name__ == "__main__":
    for split in ARQUIVOS:
        textos, rotulos = carregar_split(split)
        print(f"{split}: {len(textos)} amostras")
        for identificador, nome in CLASSES.items():
            print(f"  {identificador} {nome}: {rotulos.count(identificador)}")
