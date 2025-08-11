from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class EnergoConfig(BaseSettings):
    # AI Provider keys/models
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini")

    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20240620")

    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-70b-8192")

    # Vercel AI Gateway
    ai_gateway_api_key: Optional[str] = Field(default=None, env="AI_GATEWAY_API_KEY")
    ai_gateway_base_url: Optional[str] = Field(default="https://api.bridgit-ai.com/v1", env="AI_GATEWAY_BASE_URL")

    google_project: Optional[str] = Field(default=None, env="GOOGLE_CLOUD_PROJECT")
    google_location: str = Field(default="us-central1")

    # Agent behavior
    temperature: float = 0.3
    max_tokens: int = 1024

    # API key for protecting your API endpoints
    energo_api_key: Optional[str] = Field(default=None, env="ENERGO_API_KEY")

    # Optional: Database URL (e.g., Neon)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    # Redis for MCP connection caching, rate limiting, session management
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # CORS allowed origins (comma-separated)
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=60, env="RATE_LIMIT_REQUESTS")  # requests per minute
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")     # window in seconds

    # MCP connection management
    mcp_max_concurrency: int = Field(default=10, env="MCP_MAX_CONCURRENCY")
    mcp_idle_timeout_seconds: int = Field(default=300, env="MCP_IDLE_TIMEOUT_SECONDS")
    mcp_warm_pool_size: int = Field(default=20, env="MCP_WARM_POOL_SIZE")
    mcp_enable_cache: bool = Field(default=True, env="MCP_ENABLE_CACHE")
    mcp_cache_ttl_seconds: int = Field(default=300, env="MCP_CACHE_TTL_SECONDS")

    class Config:
        # Load from .env first (developer local), then fall back to .env.local if present (Vercel CLI default)
        env_file = (".env", ".env.local")
        extra = "ignore"
