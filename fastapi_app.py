import uuid
from typing import Any, Dict, List, Text, Union

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from yarl import URL


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
    role: Text
    content: Text


class ChatCall(BaseModel):
    sender: Text = Field(default_factory=lambda: str(uuid.uuid4()))
    system: Text
    messages: List[ChatMessage]
    metadata: Dict[Text, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    pass


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

    @app.post("/api/v1/llm/{model_name}/chat")
    async def llm_chat():
        return {"Hello": "World"}

    return app


app = create_app()
