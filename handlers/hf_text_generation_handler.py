import logging
from typing import Dict, List, Text, TypedDict, Union

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto", trust_remote_code=False
        )

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
        prompt = inputs if isinstance(inputs, Text) else inputs[0]
        print(f"Inputs: {prompt}")

        input_ids: "Tensor" = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        output: "Tensor" = self.model.generate(
            inputs=input_ids,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            max_new_tokens=2048,
        )
        input_ids = input_ids.to("cpu")
        output = output.to("cpu")

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"Generated text: {generated_text}")
        return [{"generated_text": generated_text}]

    def postprocess(self, inference_output):
        """
        Post processing of the inference output
        """

        return inference_output
