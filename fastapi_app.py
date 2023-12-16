import logging
import uuid
from typing import Any, Dict, List, Optional, Text, Union

import aiohttp
import pyjson5
from fastapi import Body, FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi import Query
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing_extensions import Literal
from yarl import URL

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_name: str = "TorchServeLlmChat"
    host_ts_inference_service: Text = "http://127.0.0.1:8085"
    host_ts_management_service: Text = "http://127.0.0.1:8086"


class ModelsResponse(BaseModel):
    models: List[Union["ModelsMeta", "ModelsMetaBasic"]]


class ModelsMetaBasic(BaseModel):
    modelName: Text
    modelUrl: Text


class ModelsMeta(ModelsMetaBasic):
    modelVersion: Text
    runtime: Text
    minWorkers: int
    maxWorkers: int
    batchSize: int
    maxBatchDelay: int
    loadedAtStartup: bool
    workers: List["ModelsWorker"]
    jobQueueStatus: Dict[Text, Any]


class ModelsWorker(BaseModel):
    id: Text
    startTime: Text
    status: Text
    memoryUsage: int
    pid: int
    gpu: bool
    gpuUsage: Text


ModelsMeta.model_rebuild()
ModelsResponse.model_rebuild()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Text


class ChatCall(BaseModel):
    sender: Text = Field(default_factory=lambda: str(uuid.uuid4()))
    system: Text = Field("")
    messages: List[ChatMessage]
    metadata: Dict[Text, Any] = Field(default_factory=dict)

    def to_prompt(self, model_name: Optional[Text] = None) -> Text:
        return self.to_system_prompt(model_name) + self.to_messages_prompt(model_name)

    def to_system_prompt(self, model_name: Optional[Text] = None) -> Text:
        sys_content = self.system.strip()
        if not sys_content:
            return ""
        sys_prompt_template = "<s>[INST] <<SYS>> {system_content} <</SYS>> [/INST]"
        return sys_prompt_template.format(system_content=sys_content)

    def to_messages_prompt(self, model_name: Optional[Text] = None) -> Text:
        msg_prompt = ""
        msg_prompt_template = "<s>[INST] {user_content} [/INST] {assistant_content}"
        for m in self.messages:
            if m.role == "user":
                msg_prompt += msg_prompt_template.format(
                    user_content=m.content.strip(), assistant_content=""
                )
            elif m.role == "assistant":
                msg_prompt += m.content.strip()
            else:
                logger.warning(f"Unknown role '{m.role}'")
        return msg_prompt


class ChatResponse(BaseModel):
    recipient_id: Text
    messages: List[ChatMessage]


class TransformersPipelineResponse(BaseModel):
    generated_text: Text


def create_app():
    app = FastAPI()
    app_settings = Settings()

    @app.get("/")
    async def root():
        return {"Hello": "World"}

    @app.get("/api/v1/llm/health")
    async def llm_health():
        url = URL(app_settings.host_ts_inference_service).with_path("ping")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()

    @app.get("/api/v1/llm/models", response_model=List[ModelsResponse])
    async def llm_models():
        url = URL(app_settings.host_ts_management_service).with_path("models")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()

    @app.get("/api/v1/llm/models/{model_name}", response_model=List[ModelsMeta])
    async def llm_model_info(model_name: Text):
        url = URL(app_settings.host_ts_management_service).with_path(
            f"models/{model_name}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                try:
                    resp.raise_for_status()
                    return await resp.json()
                except aiohttp.ClientResponseError as e:
                    if resp.status == 404:
                        raise HTTPException(
                            status_code=404, detail=f"Model '{model_name}' not found"
                        )
                    raise HTTPException(status_code=500, detail=str(e))
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/llm/predictions/{model_name}/chat", response_model=ChatResponse)
    async def llm_chat(
        model_name: Text = PathParam(..., description="Model name"),
        recipient_id: Text = Query(..., default_factory=lambda: str(uuid.uuid4())),
        chat_call: ChatCall = Body(..., description="Chat call"),
    ):
        chat_prompt = chat_call.to_prompt(model_name)
        url = URL(app_settings.host_ts_inference_service).with_path(
            f"predictions/{model_name}"
        )
        data = [{"text": chat_prompt}]
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                try:
                    resp.raise_for_status()
                    pipe_res = TransformersPipelineResponse.model_validate(
                        pyjson5.decode(await resp.text())
                    )
                    if not pipe_res.generated_text:
                        raise HTTPException(
                            status_code=500, detail="Generated text is empty"
                        )
                    pipe_res.generated_text = pipe_res.generated_text.replace(
                        chat_prompt, "", 1
                    ).strip()
                    return ChatResponse(
                        recipient_id=recipient_id,
                        messages=[
                            ChatMessage(
                                role="assistant", content=pipe_res.generated_text
                            )
                        ],
                    )
                except aiohttp.ClientResponseError as e:
                    logger.exception(e)
                    if resp.status == 404:
                        raise HTTPException(
                            status_code=404, detail=f"Model '{model_name}' not found"
                        )
                    raise HTTPException(status_code=500, detail=str(e))
                except HTTPException as e:
                    logger.exception(e)
                    raise e
                except Exception as e:
                    logger.exception(e)
                    raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()
