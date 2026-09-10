FROM python:3.12-slim

# PORT com default: o Cloud Run injeta a porta em que espera o container escutando,
# e localmente o valor de 8000 vale.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /srv

# Requirements antes do codigo: a camada de dependencias (scikit-learn e scipy somam
# uns 100 MB) so e reconstruida quando o requirements muda, nao a cada edicao de .py.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# `treino` entra na imagem porque o serving importa dele o mapa de classes e o caminho
# do artefato. `modelo` traz o joblib versionado: a imagem sobe sem treinar no build.
COPY app app
COPY treino treino
COPY modelo modelo

RUN useradd --create-home --uid 1000 api
USER api

EXPOSE 8000

# Sem curl na imagem slim, e nao vale instalar um pacote so para isso: o urllib da
# stdlib sai 0 no 200 e diferente de 0 no 503, que e a semantica desejada. A porta e
# fixa aqui porque quem usa o healthcheck e o compose local, nao o Cloud Run.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# `exec` para o uvicorn virar PID 1 e receber o SIGTERM do desligamento, em vez de o
# shell engolir o sinal e a plataforma ter que matar o container no timeout.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
