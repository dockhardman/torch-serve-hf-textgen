MODEL_DIR := models/meta-llama/Llama-2-7b-chat-hf

archive_model:
	torch-model-archiver \
		--model-name meta-llama--Llama-2-7b-chat-hf \
		--version 1.0 \
		--serialized-file $(MODEL_DIR)/model.safetensors.index.json \
		--export-path model_store \
		--handler handlers/hf_text_generation_handler.py \
		--extra-files models/meta-llama/Llama-2-7b-chat-hf/config.json,models/meta-llama/Llama-2-7b-chat-hf/generation_config.json,models/meta-llama/Llama-2-7b-chat-hf/model-00001-of-00003.safetensors,models/meta-llama/Llama-2-7b-chat-hf/model-00002-of-00003.safetensors,models/meta-llama/Llama-2-7b-chat-hf/model-00003-of-00003.safetensors,models/meta-llama/Llama-2-7b-chat-hf/model.safetensors.index.json,models/meta-llama/Llama-2-7b-chat-hf/special_tokens_map.json,models/meta-llama/Llama-2-7b-chat-hf/tokenizer.json,models/meta-llama/Llama-2-7b-chat-hf/tokenizer.model,models/meta-llama/Llama-2-7b-chat-hf/tokenizer_config.json

start_torchserve:
	torchserve --start \
		--ncs \
		--model-store model_store \
		--models meta-llama--Llama-2-7b-chat-hf=meta-llama--Llama-2-7b-chat-hf.mar \
		--ts-config config/config.properties

stop_torchserve:
	torchserve --stop

format-all:
	isort .
	black .