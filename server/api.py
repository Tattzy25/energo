from __future__ import annotations
import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator, Optional
import asyncio
from loguru import logger

from ..config import EnergoConfig
from ..core import EnergoAgent
from .security import verify_api_key

app = FastAPI(title="Energo API", version="0.1.0")

# CORS (configurable via ALLOWED_ORIGINS env, comma-separated). Defaults to allow all for dev.
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed_origins == ["*"] else allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    cfg = EnergoConfig()
    app.state.agent = EnergoAgent(cfg)
    await app.state.agent.initialize()
    logger.info("Energo API started")


@app.post("/ask", dependencies=[Depends(verify_api_key)])
async def ask_endpoint(body: AskRequest):
    agent: EnergoAgent = app.state.agent

    async def stream() -> AsyncGenerator[bytes, None]:
        async for token in agent.ask_stream(
            body.prompt,
            provider=body.provider,
            model=body.model,
            temperature=body.temperature,
        ):
            yield token.encode("utf-8")

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok"}
