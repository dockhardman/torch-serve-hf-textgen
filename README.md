# Hugging Face Transformer Model Deployment with TorchServe

This project demonstrates how to package and serve a Hugging Face transformer model using TorchServe. We use a chatbot model based on the LLaMA 2-7b model as an example.

## Project Structure

- `models/`: Contains the LLaMA 2-7b chatbot model files.
- `model_store/`: The directory where the TorchServe model archive files are stored.
- `huggingface_transformers_handler.py`: The custom handler script for processing input and output for the transformer model.
- `Makefile`: Utility to automate the packaging of the transformer model.

## Setup

1. **Install Dependencies**:
   Ensure you have Python, PyTorch, Hugging Face's Transformers library, and TorchServe installed.
2. **Model Packaging**:
   Use the Makefile to package the model. Run: `make archive_model`
3. **Start TorchServe**:
   To start the TorchServe server with the packaged model, run:
   ```bash
   torchserve --start --ncs --model-store model_store --models meta-llama--Llama-2-7b-chat-hf=meta-llama--Llama-2-7b-chat-hf.mar
   ```

## Usage

Once TorchServe is running, you can make requests to the server to get responses from the model. Example:

```bash
curl http://localhost:8080/predictions/meta-llama--Llama-2-7b-chat-hf -T sample_input.txt
```

Replace `sample_input.txt` with your input file containing the text you want the model to respond to.

## Customization

- To use a different transformer model, replace the model files in the `models/` directory and update the model name in the Makefile and TorchServe commands accordingly.
- If you need to handle different types of input or output, modify `huggingface_transformers_handler.py`.

## License

Apache License 2.0
