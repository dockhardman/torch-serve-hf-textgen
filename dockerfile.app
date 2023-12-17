FROM python:3.10-slim

LABEL maintainer="dockhardman <f1470891079@gmail.com>"

# Install System Dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    git nano vim wget curl htop ca-certificates build-essential && \
    python -m pip install --upgrade pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Install Dependencies
WORKDIR /app/
COPY ./pyproject.toml ./poetry.lock* /app/
RUN poetry install -E fastapi --no-root && poetry show

# Application
COPY ./fastapi_app.py /app/fastapi_app.py

ENTRYPOINT ["uvicorn", \
    "fastapi_app:app", \
    "--host=0.0.0.0", \
    "--port=8087", \
    "--workers=2", \
    "--reload", \
    "--log-level=debug", \
    "--use-colors", \
    "--reload-delay=5.0"]
