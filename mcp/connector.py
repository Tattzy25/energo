from __future__ import annotations
import asyncio
import time
import hashlib
import contextlib
from typing import Any, Dict, Optional, Callable
from contextlib import asynccontextmanager
from loguru import logger

from ..config.settings import EnergoConfig


class MCPConnector:
    def __init__(self, config: EnergoConfig) -> None:
        self.config = config
        # Registry of active servers: name -> connection object
        self.servers: Dict[str, Any] = {}
        # Last-used timestamp for each server
        self.last_used: Dict[str, float] = {}
        # Basic cache: key -> (expires_at, value)
        self._cache: Dict[str, tuple[float, Any]] = {}
        # In-flight coalescing: key -> Future
        self._inflight: Dict[str, asyncio.Future] = {}
        # Concurrency control
        self._sem = asyncio.Semaphore(self.config.mcp_max_concurrency)
        # Background task handle
        self._janitor_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        logger.info("Initializing MCP connectors ...")
        self.servers = {}
        self.last_used = {}
        if self._janitor_task is None:
            self._janitor_task = asyncio.create_task(self._janitor())
        logger.info("MCP initialized (on-demand)")

    async def shutdown(self) -> None:
        if self._janitor_task:
            self._janitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._janitor_task
            self._janitor_task = None
        # Close all servers
        for name in list(self.servers.keys()):
            await self._close_server(name)

    async def _janitor(self) -> None:
        # Periodically close idle connections and trim to warm pool size
        try:
            while True:
                await asyncio.sleep(5)
                await self._reap_idle()
                await self._trim_warm_pool()
        except asyncio.CancelledError:
            return

    async def _reap_idle(self) -> None:
        now = time.time()
        idle = self.config.mcp_idle_timeout_seconds
        to_close = [name for name, ts in self.last_used.items() if now - ts > idle]
        for name in to_close:
            logger.debug(f"Closing idle MCP server {name}")
            await self._close_server(name)

    async def _trim_warm_pool(self) -> None:
        # Keep only N most recently used connections
        max_keep = self.config.mcp_warm_pool_size
        if max_keep <= 0:
            # If disabled, close all
            for name in list(self.servers.keys()):
                await self._close_server(name)
            return
        # Sort by last_used desc
        alive = sorted(self.last_used.items(), key=lambda kv: kv[1], reverse=True)
        for name, _ in alive[max_keep:]:
            logger.debug(f"Trimming warm pool, closing {name}")
            await self._close_server(name)

    async def _close_server(self, name: str) -> None:
        conn = self.servers.pop(name, None)
        self.last_used.pop(name, None)
        if conn and hasattr(conn, "close"):
            try:
                res = conn.close()
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.warning(f"Error closing MCP server {name}: {e}")

    async def _get_or_create(self, server: str) -> Any:
        if server in self.servers:
            return self.servers[server]
        # On-demand: create a new connection
        conn = await self._connect_server(server)
        self.servers[server] = conn
        self.last_used[server] = time.time()
        return conn

    async def _connect_server(self, server: str) -> Any:
        # Placeholder: Replace with real MCP connection logic
        # For CLI-based servers you could spawn a process and wrap it
        logger.info(f"Connecting MCP server on-demand: {server}")
        class _Fake:
            async def echo(self, text: str) -> str:
                await asyncio.sleep(0.01)
                return text
        return _Fake()

    def _cache_key(self, server: str, method: str, kwargs: Dict[str, Any]) -> str:
        # Stable key: server + method + hashed kwargs
        h = hashlib.sha256(repr(sorted(kwargs.items())).encode()).hexdigest()
        return f"{server}:{method}:{h}"

    async def call(self, server: str, method: str, **kwargs: Any) -> Any:
        logger.debug("MCP call server={} method={} args={}", server, method, kwargs)
        cache_enabled = self.config.mcp_enable_cache
        cache_ttl = self.config.mcp_cache_ttl_seconds
        key = self._cache_key(server, method, kwargs)

        # Serve from cache if valid
        if cache_enabled:
            hit = self._cache.get(key)
            if hit and hit[0] > time.time():
                logger.debug(f"MCP cache hit for {server}.{method}")
                return hit[1]

        # Request coalescing: if an identical call in-flight, await it
        if key in self._inflight:
            logger.debug(f"MCP coalescing wait for {server}.{method}")
            return await self._inflight[key]

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._inflight[key] = fut
        try:
            async with self._sem:
                conn = await self._get_or_create(server)
                fn = getattr(conn, method, None)
                if not fn:
                    raise AttributeError(f"Method '{method}' not found on server '{server}'")
                result = await fn(**kwargs) if asyncio.iscoroutinefunction(fn) else fn(**kwargs)
                # mark last used
                self.last_used[server] = time.time()
                # cache
                if cache_enabled and cache_ttl > 0:
                    self._cache[key] = (time.time() + cache_ttl, result)
                fut.set_result(result)
                return result
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            self._inflight.pop(key, None)

    # Optional helper for explicit context usage
    @asynccontextmanager
    async def connect(self, server: str):
        conn = await self._get_or_create(server)
        try:
            yield conn
        finally:
            self.last_used[server] = time.time()
