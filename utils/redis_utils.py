from __future__ import annotations
import os
import json
import time
import asyncio
from typing import Any, Optional
from loguru import logger

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

try:
    from redis import asyncio as aioredis  # redis>=4.2
except Exception:  # pragma: no cover
    aioredis = None  # type: ignore


class RedisClient:
    """Lightweight Redis utility that supports either:
    - Upstash REST (via UPSTASH_REDIS_REST_URL/TOKEN or REDIS_REST_URL/TOKEN)
    - Standard Redis (via REDIS_URL) using redis.asyncio
    Fails open (no-ops) when not configured.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        rest_url: Optional[str] = None,
        rest_token: Optional[str] = None,
    ) -> None:
        self.redis_url = redis_url
        self.rest_url = rest_url
        self.rest_token = rest_token
        self._client = None

    @classmethod
    def from_env(cls) -> "RedisClient":
        # Prefer explicit REST creds if present
        rest_url = os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("REDIS_REST_URL")
        rest_token = os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("REDIS_REST_TOKEN")
        redis_url = os.getenv("REDIS_URL")
        return cls(redis_url=redis_url, rest_url=rest_url, rest_token=rest_token)

    def enabled(self) -> bool:
        return bool(self.redis_url or (self.rest_url and self.rest_token))

    async def _ensure_client(self):
        if self._client or not self.redis_url or not aioredis:
            return
        try:
            self._client = aioredis.from_url(self.redis_url, decode_responses=True)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to create redis asyncio client: {e}")
            self._client = None

    async def get(self, key: str) -> Optional[str]:
        if not self.enabled():
            return None
        if self.rest_url and self.rest_token:
            if not httpx:
                return None
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    # Upstash REST GET
                    resp = await client.post(
                        self.rest_url,
                        headers={"Authorization": f"Bearer {self.rest_token}"},
                        json={"commands": [["GET", key]]},
                    )
                    resp.raise_for_status()
                    out = resp.json()
                    # [[null, "value"]] on success
                    arr = out[0]
                    return arr[1] if arr and len(arr) > 1 else None
            except Exception as e:
                logger.debug(f"Redis REST GET error: {e}")
                return None
        # Standard redis
        await self._ensure_client()
        if not self._client:
            return None
        try:
            return await self._client.get(key)
        except Exception as e:  # pragma: no cover
            logger.debug(f"Redis GET error: {e}")
            return None

    async def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        if not self.enabled():
            return False
        if self.rest_url and self.rest_token:
            if not httpx:
                return False
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.post(
                        self.rest_url,
                        headers={"Authorization": f"Bearer {self.rest_token}"},
                        json={"commands": [["SETEX", key, str(ttl_seconds), value]]},
                    )
                    resp.raise_for_status()
                    return True
            except Exception as e:
                logger.debug(f"Redis REST SETEX error: {e}")
                return False
        await self._ensure_client()
        if not self._client:
            return False
        try:
            await self._client.setex(key, ttl_seconds, value)
            return True
        except Exception as e:  # pragma: no cover
            logger.debug(f"Redis SETEX error: {e}")
            return False

    async def incr_with_expiry(self, key: str, window_seconds: int) -> int:
        """Atomic increment with expiry for rate-limiting counters."""
        if not self.enabled():
            return 0
        if self.rest_url and self.rest_token:
            if not httpx:
                return 0
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    # Pipeline: INCR key; EXPIRE key window NX
                    resp = await client.post(
                        self.rest_url,
                        headers={"Authorization": f"Bearer {self.rest_token}"},
                        json={
                            "commands": [
                                ["INCR", key],
                                ["EXPIRE", key, str(window_seconds), "NX"],
                            ]
                        },
                    )
                    resp.raise_for_status()
                    out = resp.json()
                    # First result is incr value: [null, 1]
                    return int(out[0][1]) if out and out[0] and out[0][1] is not None else 0
            except Exception as e:
                logger.debug(f"Redis REST INCR error: {e}")
                return 0
        await self._ensure_client()
        if not self._client:
            return 0
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_seconds, nx=True)
            res = await pipe.execute()
            return int(res[0]) if res else 0
        except Exception as e:  # pragma: no cover
            logger.debug(f"Redis INCR error: {e}")
            return 0

    async def acquire_lock(self, key: str, ttl_ms: int, token: Optional[str] = None) -> Optional[str]:
        """Best-effort distributed lock. Returns lock token if acquired."""
        token = token or str(os.urandom(16).hex())
        if not self.enabled():
            return token
        if self.rest_url and self.rest_token:
            if not httpx:
                return None
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    # SET key token NX PX ttl
                    resp = await client.post(
                        self.rest_url,
                        headers={"Authorization": f"Bearer {self.rest_token}"},
                        json={"commands": [["SET", key, token, "NX", "PX", str(ttl_ms)]]},
                    )
                    resp.raise_for_status()
                    out = resp.json()
                    ok = out and out[0] and out[0][1] == "OK"
                    return token if ok else None
            except Exception as e:
                logger.debug(f"Redis REST LOCK error: {e}")
                return None
        await self._ensure_client()
        if not self._client:
            return token
        try:
            ok = await self._client.set(key, token, nx=True, px=ttl_ms)
            return token if ok else None
        except Exception as e:
            logger.debug(f"Redis LOCK error: {e}")
            return None

    async def release_lock(self, key: str, token: str) -> bool:
        if not self.enabled():
            return True
        # Best-effort: check value matches then DEL
        if self.rest_url and self.rest_token:
            if not httpx:
                return False
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    # GET then conditional DEL (race-prone but acceptable for non-critical paths)
                    resp = await client.post(
                        self.rest_url,
                        headers={"Authorization": f"Bearer {self.rest_token}"},
                        json={"commands": [["GET", key]]},
                    )
                    resp.raise_for_status()
                    out = resp.json()
                    cur = out[0][1] if out and out[0] else None
                    if cur == token:
                        await client.post(
                            self.rest_url,
                            headers={"Authorization": f"Bearer {self.rest_token}"},
                            json={"commands": [["DEL", key]]},
                        )
                        return True
                    return False
            except Exception as e:
                logger.debug(f"Redis REST UNLOCK error: {e}")
                return False
        await self._ensure_client()
        if not self._client:
            return True
        try:
            cur = await self._client.get(key)
            if cur == token:
                await self._client.delete(key)
                return True
            return False
        except Exception as e:
            logger.debug(f"Redis UNLOCK error: {e}")
            return False