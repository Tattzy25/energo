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
    ai_gateway_base_url: Optional[str] = Field(default="https://ai-gateway.vercel.sh/v1", env="AI_GATEWAY_BASE_URL")

    google_project: Optional[str] = Field(default=None, env="GOOGLE_CLOUD_PROJECT")
    google_location: str = Field(default="us-central1")

    # Agent behavior
    temperature: float = 0.3
    max_tokens: int = 1024

    # API key for protecting your API endpoints
    energo_api_key: Optional[str] = Field(default=None, env="ENERGO_API_KEY")

    # Optional: Database URL (e.g., Neon)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"
