import torch
import transformers
from transformers import AutoTokenizer
from transformers.pipelines import TextGenerationPipeline
from ts.torch_handler.base_handler import BaseHandler


class TransformersHandler(BaseHandler):
    """
    Transformers handler class for text generation.
    """

    def __init__(self):
        super(TransformersHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        """
        Initialize the model. This will be called during model loading.
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.pipeline: "TextGenerationPipeline" = transformers.pipeline(
            "text-generation",
            model=model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = self.pipeline.model

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocessing input request. This prepares the input text.
        """
        texts = [request.get("data") or request.get("body") for request in data]
        texts = [
            text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else text
            for text in texts
        ]
        return texts

    def inference(self, inputs):
        """
        Generate text using the model pipeline.
        """
        outputs = self.pipeline(
            inputs,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=4096,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
        )
        return [output["generated_text"] for output in outputs]

    def postprocess(self, inference_output):
        """
        Post processing of the inference output
        """
        return inference_output
