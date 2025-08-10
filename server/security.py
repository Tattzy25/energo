from __future__ import annotations
from fastapi import Header, HTTPException
from typing import Optional
from ..config import EnergoConfig


async def verify_api_key(x_energo_key: Optional[str] = Header(default=None)) -> None:
    """Simple header-based API key auth. Set ENERGO_API_KEY in environment.
    If no key is configured, auth is skipped (useful for local dev).
    """
    cfg = EnergoConfig()
    if not cfg.energo_api_key:
        return
    if x_energo_key != cfg.energo_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
