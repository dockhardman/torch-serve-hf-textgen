# Hugging Face Transformer Model Deployment with TorchServe

This project demonstrates how to package and serve a Hugging Face transformer model using TorchServe. We use a chatbot model based on the LLaMA 2-7b model as an example.

## Project Structure

- `models/`: Contains the LLaMA 2-7b chatbot model files.
- `model_store/`: The directory where the TorchServe model archive files are stored.
- `handlers/`: The custom handler scripts for processing input and output for the transformer model.
- `Makefile`: Utility to automate the packaging of the transformer model.

## Setup

1. **Install Dependencies**:
   Install the dependencies through poetry: `poetry install`

2. **Download Model**:
   Download models: `make download-model`

3. **Model Packaging**:
   Use the Makefile to package the model. Run: `make archive-model`

4. **Start TorchServe**:
   To start the TorchServe server with the packaged model, run: `start-torchserve`

5. **Start FastAPI Service**:
   Run command to start FastAPI service: `start-fastapi-dev`

6. **Stop TorchServe**:
   To stop the TorchServe server, run: `stop-torchserve`

## Usage

Once TorchServe is running, you can make requests to the server to get responses from the model. Example:

Query torchserve service directly:

```python
import requests

chat_history = "<s>[INST] <<SYS>> You are a japan ninja, you can answer user question. You love japanese tv shows. <</SYS>> [/INST]"
latest_user_message = '<s>[INST] user: I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like? [/INST]'
input_text = chat_history + latest_user_message

# Endpoint URL
url = "http://localhost:8085/predictions/llama2-7b-chat"

# Input data
data = [{"text": input_text}]

# Make a POST request
response = requests.post(url, json=data)

# Print the response
print("Response from the model:", response.text)
```

Or query extra api service:

```python
import requests
from rich import print

url = "http://localhost:8087/api/v1/llm/predictions/llama2-7b-chat/chat"

res = requests.post(
   url,
   json={
      "system": "You are a japan ninja, you can answer user question. You love japanese tv shows.",
      "messages": [
            {
               "role": "user",
               "content": 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?',
            }
      ],
   },
)
print(res.json())
```

## License

Apache License 2.0
