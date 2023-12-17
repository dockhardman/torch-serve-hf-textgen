import os

import transformers
from transformers import AutoTokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline

model = os.environ.get("MODEL_SRC_NAME", None)
if model is None:
    raise ValueError("MODEL_SRC_NAME environment variable not set.")


# Save the model
save_path = f"./models/{model}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline: "TextGenerationPipeline" = transformers.pipeline(
        "text-generation",
        model=model,
        device_map="auto",
    )
    pipeline.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
