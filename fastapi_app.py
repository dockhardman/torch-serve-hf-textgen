import uuid
from typing import Any, Dict, List, Text

import aiohttp
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from yarl import URL


class Settings(BaseSettings):
    app_name: str = "TorchServeLlmChat"
    host_ts_service: Text = "http://127.0.0.1:8085"


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
        url = URL(app_settings.host_ts_service).with_path("ping")
        print(url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()

    @app.post("/api/v1/llm/{model_name}/chat")
    async def llm_chat():
        return {"Hello": "World"}

    return app


app = create_app()
