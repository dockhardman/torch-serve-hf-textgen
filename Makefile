MODEL_NAME := llama2-7b-chat
MODEL_SRC_NAME := TheBloke/Llama-2-7b-Chat-GPTQ
MODEL_DIR := models/TheBloke/Llama-2-7b-Chat-GPTQ
MODEL_EXTRA_FILES := $(shell find $(MODEL_DIR) -type f | paste -sd, -)

PYTHONPATH ?= $(CURDIR)


download-model:
	PYTHONPATH=$(PYTHONPATH) python scripts/download_model.py

archive-model:
	torch-model-archiver \
		--model-name $(MODEL_NAME) \
		--version 1.0 \
		--serialized-file $(MODEL_DIR)/model.safetensors \
		--export-path model_store \
		--handler handlers/hf_text_generation_handler.py \
		--extra-files $(MODEL_EXTRA_FILES)

start-torchserve:
	torchserve --start \
		--ncs \
		--model-store model_store \
		--models $(MODEL_NAME)=$(MODEL_NAME).mar \
		--ts-config config/config.properties

stop-torchserve:
	torchserve --stop

start-fastapi-dev:
	uvicorn fastapi_app:app \
		--host=0.0.0.0 \
		--port=8087 \
		--workers=2 \
		--reload \
		--log-level=debug \
		--use-colors \
		--reload-delay=5.0

start-fastapi:
	uvicorn fastapi_app:app \
		--host=0.0.0.0 \
		--port=80 \
		--workers=2

format-all:
	isort .
	black .
