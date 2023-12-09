MODEL_DIR := models/meta-llama/Llama-2-7b-chat-hf
EXTRA_FILES := $(shell for f in $(MODEL_DIR)/*; do echo $$f,; done | sed 's/,$$/ /')


archive_model:
	torch-model-archiver \
		--model-name meta-llama--Llama-2-7b-chat-hf \
		--version 1.0 \
		--serialized-file $(MODEL_DIR)/model.safetensors.index.json \
		--export-path model_store \
		--handler huggingface_transformers_handler.py \
		--extra-files "$(EXTRA_FILES)"

start_torchserve:
	torchserve --start \
		--ncs \
		--model-store model_store \
		--models meta-llama--Llama-2-7b-chat-hf=meta-llama--Llama-2-7b-chat-hf.mar

stop_torchserve:
	torchserve --stop

format-all:
	isort .
	black .
