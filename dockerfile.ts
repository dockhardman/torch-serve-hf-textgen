FROM pytorch/torchserve:latest-gpu

ENV MODEL_NAME=llama2-7b-chat

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Install Dependencies
WORKDIR /home/model-server/
COPY ./pyproject.toml ./poetry.lock* /home/model-server/
RUN poetry install -E ts --no-root && poetry show

# Copy model artifacts, custom handler and other dependencies
COPY ./model_store/llama2-7b-chat.mar /home/model-server/model_store/

EXPOSE 8085
EXPOSE 8086

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
    "--start", \
    "--ts-config=/home/model-server/config/config.properties", \
    "--models", \
    "$MODEL_NAME=$MODEL_NAME.mar", \
    "--model-store", \
    "/home/model-server/model_store"]
