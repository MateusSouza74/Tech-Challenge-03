# Tech Challenge 03: classificação de laudos médicos

Classificador de texto clínico servido por API, empacotado em container, com pipeline de
treino orquestrado, CI automatizado e stack de observabilidade. O caso de uso é triagem:
o laudo chega, e a resposta com a categoria da condição volta na mesma sessão do usuário.

## Dataset

[Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus):
14.438 abstracts médicos rotulados em 5 classes de condição do paciente. Bem acima do
mínimo de 2.000 amostras pedido pelo desafio, e com o split de treino e teste já definido
pelos autores, o que evita discussão sobre vazamento de dados na avaliação.

| Classe | Treino | Teste | Total |
|---|---:|---:|---:|
| neoplasms | 2.530 | 633 | 3.163 |
| digestive system diseases | 1.195 | 299 | 1.494 |
| nervous system diseases | 1.540 | 385 | 1.925 |
| cardiovascular diseases | 2.441 | 610 | 3.051 |
| general pathological conditions | 3.844 | 961 | 4.805 |
| **Total** | **11.550** | **2.888** | **14.438** |

A classe majoritária concentra 33% das amostras, e é esse o piso contra o qual a acurácia
do modelo tem que ser lida.

## Etapas do desafio

| Etapa | Entrega | Status |
|---|---|---|
| 1 | Decisão arquitetural de nuvem, API FastAPI, container e baseline de latência | em andamento |
| 2 | GitHub Actions com lint e testes, DAG de treino no Airflow | a fazer |
| 3 | Docker Compose com Prometheus e Grafana, dashboard de métricas | a fazer |
| 4 | Otimização de latência do modelo e comparação com o baseline | a fazer |

## Desenvolvimento

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux e macOS: source .venv/bin/activate
pip install -r requirements-dev.txt

ruff check .
pytest
```
