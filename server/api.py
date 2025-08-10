from __future__ import annotations
import os
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, Dict, Any
import asyncio
from loguru import logger
import time
import uuid
import secrets
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from ..config import EnergoConfig
from ..core import EnergoAgent
from .security import verify_api_key
from ..db.session import get_db
from ..db import models as dbm

app = FastAPI(title="Energo API", version="0.1.0")

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    with logger.contextualize(request_id=rid, path=request.url.path, method=request.method):
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

# CORS (configurable via ALLOWED_ORIGINS env, comma-separated). Defaults to allow all for dev.
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
normalized_origins = ["*"] if allowed_origins == ["*"] else [
    o if o.startswith("http://") or o.startswith("https://") else f"https://{o}"
    for o in allowed_origins
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=normalized_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limiter (per-IP)
_rate_state = {"window_start": 0.0, "counts": {}}

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    cfg = EnergoConfig()
    max_req = cfg.rate_limit_requests
    window = cfg.rate_limit_window
    if max_req <= 0 or window <= 0:
        return await call_next(request)
    now = time.time()
    # reset window if elapsed
    if now - _rate_state["window_start"] > window:
        _rate_state["window_start"] = now
        _rate_state["counts"] = {}
    # key by IP and path
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{request.url.path}"
    _rate_state["counts"][key] = _rate_state["counts"].get(key, 0) + 1
    if _rate_state["counts"][key] > max_req:
        logger.warning(f"Rate limit exceeded for {key}")
        return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})
    return await call_next(request)


class AskRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    cfg = EnergoConfig()
    app.state.agent = EnergoAgent(cfg)
    
    # Ensure DB schema exists (idempotent)
    try:
        from ..db.models import Base
        from ..db.session import engine
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logger.warning(f"DB schema initialization skipped/failed: {e}")

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

# Settings documentation for UI or introspection
_SETTINGS_DOCS: Dict[str, str] = {
    "openai_api_key": "API key for OpenAI (server-side only).",
    "openai_model": "Default OpenAI model to use.",
    "anthropic_api_key": "API key for Anthropic (server-side only).",
    "anthropic_model": "Default Anthropic model to use.",
    "groq_api_key": "API key for Groq (server-side only).",
    "groq_model": "Default Groq model to use.",
    "ai_gateway_api_key": "API key for Vercel AI Gateway (server-side only).",
    "ai_gateway_base_url": "Base URL for Vercel AI Gateway (e.g., https://ai-gateway.vercel.sh/v1).",
    "google_project": "GCP project for Vertex AI.",
    "google_location": "GCP region for Vertex AI (e.g., us-central1).",
    "temperature": "Sampling temperature for LLMs (0=deterministic, higher=more random).",
    "max_tokens": "Max tokens per response.",
    "energo_api_key": "Protects /ask endpoint (set to require x-energo-key header).",
    "database_url": "Database connection string (e.g., Neon).",
    "redis_url": "Redis connection URL for caching/limits/locks (optional).",
    "allowed_origins": "Comma-separated CORS origins (hostname or URL).",
    "rate_limit_requests": "Max requests per window (per IP, per path). Set 0 to disable.",
    "rate_limit_window": "Window size in seconds for rate limiting.",
    "mcp_max_concurrency": "Max concurrent MCP calls (bounded semaphore).",
    "mcp_idle_timeout_seconds": "Close MCP connections idle longer than this.",
    "mcp_warm_pool_size": "Keep most recently used MCP connections hot up to this size.",
    "mcp_enable_cache": "Enable in-memory/Redis cache for deterministic MCP results.",
    "mcp_cache_ttl_seconds": "TTL in seconds for MCP cache entries.",
}

_SECRET_KEYS = {"openai_api_key", "anthropic_api_key", "groq_api_key", "ai_gateway_api_key", "energo_api_key", "database_url", "redis_url"}


def _redact(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    if len(s) <= 6:
        return "***"
    return s[:3] + "***" + s[-3:]


@app.get("/settings")
async def get_settings():
    cfg = EnergoConfig()
    # Build current settings view with redaction
    raw: Dict[str, Any] = {
        "openai_api_key": cfg.openai_api_key,
        "openai_model": cfg.openai_model,
        "anthropic_api_key": cfg.anthropic_api_key,
        "anthropic_model": cfg.anthropic_model,
        "groq_api_key": cfg.groq_api_key,
        "groq_model": cfg.groq_model,
        "ai_gateway_api_key": cfg.ai_gateway_api_key,
        "ai_gateway_base_url": cfg.ai_gateway_base_url,
        "google_project": cfg.google_project,
        "google_location": cfg.google_location,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "energo_api_key": cfg.energo_api_key,
        "database_url": cfg.database_url,
        "redis_url": cfg.redis_url,
        "allowed_origins": cfg.allowed_origins,
        "rate_limit_requests": cfg.rate_limit_requests,
        "rate_limit_window": cfg.rate_limit_window,
        "mcp_max_concurrency": cfg.mcp_max_concurrency,
        "mcp_idle_timeout_seconds": cfg.mcp_idle_timeout_seconds,
        "mcp_warm_pool_size": cfg.mcp_warm_pool_size,
        "mcp_enable_cache": cfg.mcp_enable_cache,
        "mcp_cache_ttl_seconds": cfg.mcp_cache_ttl_seconds,
    }
    current = {k: (_redact(v) if k in _SECRET_KEYS else v) for k, v in raw.items()}
    return {"settings": current, "descriptions": _SETTINGS_DOCS}


class ToolCreate(BaseModel):
    name: str
    display_name: str | None = None
    description: str | None = None
    category: str | None = None
    version: str | None = "1.0.0"
    endpoint_url: str | None = None
    connection_type: str | None = "websocket"
    pricing_model: str | None = "free"
    price_per_call: float | None = 0.0
    rate_limit_per_hour: int | None = 1000
    tags: dict | list | None = None
    config_schema: dict | None = None
    is_active: bool | None = True
    is_public: bool | None = True


@app.get("/catalog")
async def list_tools(public_only: bool = True, db: Session = Depends(get_db)):
    q = db.query(dbm.MCPTool)
    if public_only:
        q = q.filter(dbm.MCPTool.is_public == True, dbm.MCPTool.is_active == True)
    tools = q.order_by(dbm.MCPTool.name.asc()).all()
    return [{
        "id": t.id,
        "name": t.name,
        "display_name": t.display_name,
        "description": t.description,
        "category": t.category,
        "version": t.version,
        "endpoint_url": t.endpoint_url,
        "connection_type": t.connection_type,
        "pricing_model": t.pricing_model,
        "price_per_call": t.price_per_call,
        "rate_limit_per_hour": t.rate_limit_per_hour,
        "tags": t.tags,
        "health_status": t.health_status,
        "updated_at": t.updated_at,
    } for t in tools]


@app.post("/admin/tools", dependencies=[Depends(verify_api_key)])
async def create_tool(payload: ToolCreate, db: Session = Depends(get_db)):
    exists = db.query(dbm.MCPTool).filter(dbm.MCPTool.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=400, detail="Tool with that name already exists")
    tool = dbm.MCPTool(
        name=payload.name,
        display_name=payload.display_name,
        description=payload.description,
        category=payload.category,
        version=payload.version,
        endpoint_url=payload.endpoint_url,
        connection_type=payload.connection_type,
        pricing_model=payload.pricing_model,
        price_per_call=payload.price_per_call or 0.0,
        rate_limit_per_hour=payload.rate_limit_per_hour or 1000,
        tags=payload.tags,
        config_schema=payload.config_schema,
        is_active=True if payload.is_active is None else payload.is_active,
        is_public=True if payload.is_public is None else payload.is_public,
        health_status="unknown",
    )
    db.add(tool)
    db.commit()
    db.refresh(tool)
    return {"id": tool.id, "name": tool.name}


class IssueSessionRequest(BaseModel):
    tool_name: str
    client_id: str | None = None
    ttl_seconds: int | None = 900


@app.post("/admin/sessions/issue", dependencies=[Depends(verify_api_key)])
async def issue_session_token(body: IssueSessionRequest, request: Request, db: Session = Depends(get_db)):
    tool = db.query(dbm.MCPTool).filter(dbm.MCPTool.name == body.tool_name, dbm.MCPTool.is_active == True).first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found or inactive")
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    expires = now + timedelta(seconds=body.ttl_seconds or 900)
    sess = dbm.MCPSession(
        session_token=token,
        tool_id=tool.id,
        client_id=body.client_id,
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        created_at=now,
        expires_at=expires,
        is_active=True,
    )
    db.add(sess)
    db.commit()
    return {"session_token": token, "tool": tool.name, "expires_at": expires.isoformat()}


class UsageLogIn(BaseModel):
    session_token: str
    tool_name: str
    method_name: str
    status_code: int
    duration_ms: int | None = None
    request_size_bytes: int | None = None
    response_size_bytes: int | None = None
    error_message: str | None = None
    billable_units: float | None = 1.0
    cost_usd: float | None = 0.0


@app.post("/usage")
async def record_usage(payload: UsageLogIn, db: Session = Depends(get_db)):
    # Lightweight auth for now: require a valid active session token
    sess = db.query(dbm.MCPSession).filter(dbm.MCPSession.session_token == payload.session_token, dbm.MCPSession.is_active == True).first()
    if not sess:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    tool = db.query(dbm.MCPTool).filter(dbm.MCPTool.name == payload.tool_name).first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    log = dbm.MCPUsageLog(
        session_id=sess.id,
        tool_id=tool.id,
        method_name=payload.method_name,
        status_code=payload.status_code,
        duration_ms=payload.duration_ms,
        request_size_bytes=payload.request_size_bytes or 0,
        response_size_bytes=payload.response_size_bytes or 0,
        error_message=payload.error_message,
        billable_units=payload.billable_units or 1.0,
        cost_usd=payload.cost_usd or 0.0,
    )
    db.add(log)
    # Simple counters on session
    sess.calls_made = (sess.calls_made or 0) + 1
    db.commit()
    return {"ok": True}
