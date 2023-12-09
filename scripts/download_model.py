import torch
import transformers
from transformers import AutoTokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline

model = "meta-llama/Llama-2-7b-chat-hf"


# Save the model
save_path = f"./models/{model}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline: "TextGenerationPipeline" = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipeline.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
