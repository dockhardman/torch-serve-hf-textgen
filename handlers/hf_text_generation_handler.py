import logging
from typing import Dict, List, Text, TypedDict, Union

import torch
import transformers
from transformers import AutoTokenizer
from transformers.pipelines import TextGenerationPipeline
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TransformersHandler(BaseHandler):
    """
    Transformers handler class for text generation.
    """

    class RequestCall(TypedDict):
        body: Union[
            "TransformersHandler.RequestBody", List["TransformersHandler.RequestBody"]
        ]

    class RequestBody(TypedDict):
        text: Text

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
            # torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model = self.pipeline.model

        self.initialized = True

    def preprocess(self, data: List["RequestCall"]) -> List[Text]:
        """
        Preprocessing input request. This prepares the input text.
        """
        if not data:
            return []
        request_body = data[0].get("data") or data[0].get("body")
        if isinstance(request_body, Dict):
            request_body = [request_body]
        texts: List[Text] = []
        for request in request_body:
            text = request.get("text")
            if text is None:
                continue
            texts.append(
                text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else text
            )
        print(f"Texts: {texts}")
        return texts

    def inference(self, inputs: List[Text]) -> List[Text]:
        """
        Generate text using the model pipeline.
        """
        if not inputs:
            return []
        if isinstance(inputs, Text):
            inputs = [inputs]
        print(f"Inputs: {inputs}")
        results = self.pipeline(
            inputs,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=4096,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
        )
        print(f"Results: {results}")
        return [result[0] for result in results]

    def postprocess(self, inference_output):
        """
        Post processing of the inference output
        """
        return inference_output
