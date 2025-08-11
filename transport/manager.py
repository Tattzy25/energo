from __future__ import annotations
import asyncio
import json
import time
import random
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from loguru import logger
# Removed direct websockets imports to avoid import-time errors on incompatible environments
from sqlalchemy.orm import Session

from ..config import EnergoConfig
from ..db.models import MCPTool
from ..db.session import SessionLocal
from ..utils.redis_utils import RedisClient


@dataclass
class HealthStatus:
    status: str  # "healthy", "degraded", "down"
    last_check: float
    consecutive_failures: int = 0
    avg_response_time_ms: float = 0.0


@dataclass
class ConnectionPool:
    server: str
    endpoint: str
    max_connections: int = 5
    active_connections: List[Any] = field(default_factory=list)
    idle_connections: List[Any] = field(default_factory=list)
    pending_requests: int = 0
    health: HealthStatus = field(default_factory=lambda: HealthStatus("unknown", 0))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Transport(Protocol):
    """Protocol for different MCP transport types."""
    
    async def connect(self, endpoint: str) -> Any:
        """Establish connection to MCP server."""
        ...
    
    async def call(self, connection: Any, method: str, params: dict) -> Any:
        """Make RPC call over the transport."""
        ...
    
    async def close(self, connection: Any) -> None:
        """Close the connection."""
        ...


class WebSocketTransport:
    """WebSocket transport for MCP servers with lazy imports."""
    
    async def connect(self, endpoint: str) -> Any:
        """Connect to WebSocket MCP server."""
        try:
            import websockets as ws_lib
        except ImportError as e:
            raise RuntimeError("websockets package is not installed. Please add 'websockets' to your dependencies.") from e
        try:
            ws = await ws_lib.connect(
                endpoint,
                timeout=10,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            logger.debug(f"WebSocket connected to {endpoint}")
            return ws
        except Exception as e:
            logger.error(f"WebSocket connection failed to {endpoint}: {e}")
            raise
    
    async def call(self, ws: Any, method: str, params: dict) -> Any:
        """Make JSON-RPC call over WebSocket."""
        try:
            import websockets as ws_lib
        except ImportError as e:
            raise RuntimeError("websockets package is not installed. Please add 'websockets' to your dependencies.") from e

        request_id = random.randint(1, 1000000)
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        try:
            await ws.send(json.dumps(message))
            response = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(response)
            
            if "error" in data:
                raise RuntimeError(f"MCP error: {data['error']}")
            
            return data.get("result")
        except asyncio.TimeoutError:
            raise RuntimeError("MCP call timeout")
        except ws_lib.exceptions.ConnectionClosed:
            raise RuntimeError("MCP connection closed")
    
    async def close(self, ws: Any) -> None:
        """Close WebSocket connection."""
        try:
            # Import lazily to avoid hard dependency at import time
            import websockets as ws_lib  # noqa: F401
            await ws.close()
        except Exception as e:
            logger.debug(f"Error closing WebSocket: {e}")


class TransportManager:
    """Manages connections to MCP servers with health checks, pooling, and backoff."""
    
    def __init__(self, config: EnergoConfig, redis_client: Optional[RedisClient] = None):
        self.config = config
        self.redis = redis_client or RedisClient.from_env()
        self.pools: Dict[str, ConnectionPool] = {}
        self.transports = {
            "websocket": WebSocketTransport()
        }
        self._health_check_task: Optional[asyncio.Task] = None
        self._pool_manager_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Start background tasks for health checks and pool management."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        if self._pool_manager_task is None:
            self._pool_manager_task = asyncio.create_task(self._pool_manager_loop())
        logger.info("Transport manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown transport manager and close all connections."""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._pool_manager_task:
            self._pool_manager_task.cancel()
        
        # Close all connection pools
        for pool in self.pools.values():
            await self._close_pool(pool)
        
        logger.info("Transport manager shutdown complete")
    
    async def call(self, tool_name: str, method: str, params: dict = None) -> Any:
        """Make a call to an MCP tool with smart routing and resilience."""
        # Get tool info from database
        db = SessionLocal()
        try:
            tool = db.query(MCPTool).filter(
                MCPTool.name == tool_name,
                MCPTool.is_active == True
            ).first()
            
            if not tool:
                raise RuntimeError(f"Tool '{tool_name}' not found or inactive")
            
            if not tool.endpoint_url:
                raise RuntimeError(f"Tool '{tool_name}' has no endpoint configured")
            
            # Use connection pool for the call
            return await self._call_with_pool(tool, method, params or {})
        
        finally:
            db.close()
    
    async def _call_with_pool(self, tool: MCPTool, method: str, params: dict) -> Any:
        """Make call using connection pool with exponential backoff."""
        pool_key = f"{tool.name}:{tool.endpoint_url}"
        
        # Get or create pool
        if pool_key not in self.pools:
            self.pools[pool_key] = ConnectionPool(
                server=tool.name,
                endpoint=tool.endpoint_url,
                max_connections=3  # Conservative default
            )
        
        pool = self.pools[pool_key]
        
        # Health check - skip if server is down
        if pool.health.status == "down":
            time_since_check = time.time() - pool.health.last_check
            # Exponential backoff: 30s, 60s, 120s, 300s max
            backoff_delay = min(300, 30 * (2 ** pool.health.consecutive_failures))
            if time_since_check < backoff_delay:
                raise RuntimeError(f"Server {tool.name} is down, backing off for {backoff_delay}s")
        
        # Acquire connection from pool
        async with pool.lock:
            connection = await self._get_connection(pool, tool.connection_type)
        
        try:
            start_time = time.time()
            
            # Make the call
            transport = self.transports.get(tool.connection_type)
            if not transport:
                raise RuntimeError(f"Unsupported connection type: {tool.connection_type}")
            
            result = await transport.call(connection, method, params)
            
            # Update health metrics
            duration_ms = (time.time() - start_time) * 1000
            await self._update_health_success(pool, duration_ms)
            
            return result
        
        except Exception as e:
            await self._update_health_failure(pool)
            raise RuntimeError(f"MCP call failed: {e}")
        
        finally:
            # Return connection to pool
            async with pool.lock:
                if len(pool.idle_connections) < 2:  # Keep some connections idle
                    pool.idle_connections.append(connection)
                else:
                    await self._close_connection(connection, tool.connection_type)
    
    async def _get_connection(self, pool: ConnectionPool, connection_type: str) -> Any:
        """Get or create a connection from the pool."""
        # Try to reuse idle connection
        if pool.idle_connections:
            return pool.idle_connections.pop()
        
        # Create new connection if under limit
        if len(pool.active_connections) < pool.max_connections:
            transport = self.transports[connection_type]
            connection = await transport.connect(pool.endpoint)
            pool.active_connections.append(connection)
            return connection
        
        # Pool exhausted - wait or fail
        raise RuntimeError(f"Connection pool exhausted for {pool.server}")
    
    async def _close_connection(self, connection: Any, connection_type: str) -> None:
        """Close a single connection."""
        try:
            transport = self.transports[connection_type]
            await transport.close(connection)
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")
    
    async def _update_health_success(self, pool: ConnectionPool, duration_ms: float) -> None:
        """Update health status after successful call."""
        pool.health.status = "healthy"
        pool.health.last_check = time.time()
        pool.health.consecutive_failures = 0
        
        # Update average response time (exponential moving average)
        if pool.health.avg_response_time_ms == 0:
            pool.health.avg_response_time_ms = duration_ms
        else:
            pool.health.avg_response_time_ms = (
                0.7 * pool.health.avg_response_time_ms + 0.3 * duration_ms
            )
        
        # Cache health status in Redis
        if self.redis.enabled():
            health_key = f"mcp:health:{pool.server}"
            health_data = {
                "status": pool.health.status,
                "last_check": pool.health.last_check,
                "avg_response_time_ms": pool.health.avg_response_time_ms
            }
            await self.redis.setex(health_key, 300, json.dumps(health_data))
    
    async def _update_health_failure(self, pool: ConnectionPool) -> None:
        """Update health status after failed call."""
        pool.health.consecutive_failures += 1
        pool.health.last_check = time.time()
        
        if pool.health.consecutive_failures >= 3:
            pool.health.status = "down"
        elif pool.health.consecutive_failures >= 1:
            pool.health.status = "degraded"
        
        logger.warning(f"Health failure for {pool.server}: {pool.health.consecutive_failures} consecutive failures")
    
    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _run_health_checks(self) -> None:
        """Run health checks on all pools."""
        for pool in self.pools.values():
            try:
                # Simple ping check
                async with pool.lock:
                    if pool.idle_connections:
                        connection = pool.idle_connections[0]
                        # Try a simple method call as health check
                        transport = self.transports.get("websocket")  # Default
                        if transport:
                            start_time = time.time()
                            await transport.call(connection, "ping", {})
                            duration_ms = (time.time() - start_time) * 1000
                            await self._update_health_success(pool, duration_ms)
            except Exception:
                await self._update_health_failure(pool)
    
    async def _pool_manager_loop(self) -> None:
        """Background task for managing connection pools."""
        while True:
            try:
                await asyncio.sleep(30)  # Manage every 30 seconds
                await self._manage_pools()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pool manager loop error: {e}")
    
    async def _manage_pools(self) -> None:
        """Clean up idle connections and resize pools."""
        for pool in list(self.pools.values()):
            async with pool.lock:
                # Close excess idle connections
                while len(pool.idle_connections) > 2:
                    connection = pool.idle_connections.pop()
                    await self._close_connection(connection, "websocket")
                
                # Remove pools with no activity
                if (len(pool.active_connections) == 0 and 
                    len(pool.idle_connections) == 0 and 
                    time.time() - pool.health.last_check > 3600):
                    # Remove inactive pools after 1 hour
                    pool_key = f"{pool.server}:{pool.endpoint}"
                    if pool_key in self.pools:
                        del self.pools[pool_key]
                    logger.debug(f"Removed inactive pool: {pool.server}")
    
    async def _close_pool(self, pool: ConnectionPool) -> None:
        """Close all connections in a pool."""
        async with pool.lock:
            for connection in pool.active_connections + pool.idle_connections:
                await self._close_connection(connection, "websocket")
            pool.active_connections.clear()
            pool.idle_connections.clear()
    
    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current health status of all pools."""
        status = {}
        for pool in self.pools.values():
            status[pool.server] = {
                "status": pool.health.status,
                "last_check": pool.health.last_check,
                "consecutive_failures": pool.health.consecutive_failures,
                "avg_response_time_ms": pool.health.avg_response_time_ms,
                "active_connections": len(pool.active_connections),
                "idle_connections": len(pool.idle_connections)
            }
        return status