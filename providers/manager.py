from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
from loguru import logger

from .providers import BaseProvider, OpenAIProvider, AnthropicProvider, GroqProvider, GoogleVertexProvider, AIGatewayProvider
from ..config.settings import EnergoConfig


class AIProviderManager:
    def __init__(self, config: EnergoConfig) -> None:
        self.config = config
        self.providers: Dict[str, BaseProvider] = {}

    async def initialize(self) -> None:
        logger.info("Initializing AI providers ...")
        # Initialize based on available keys
        # Priority: AI Gateway first (if configured), then direct providers
        if self.config.ai_gateway_api_key:
            self.providers["ai-gateway"] = AIGatewayProvider(self.config)
        if self.config.openai_api_key:
            self.providers["openai"] = OpenAIProvider(self.config)
        if self.config.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(self.config)
        if self.config.groq_api_key:
            self.providers["groq"] = GroqProvider(self.config)
        if self.config.google_project:
            self.providers["google_vertex"] = GoogleVertexProvider(self.config)
        logger.info("Providers ready: {}", list(self.providers.keys()))

    async def generate(self, prompt: str, provider: Optional[str] = None, **kwargs: Any) -> str:
        if provider:
            p = self.providers.get(provider)
            if not p:
                raise ValueError(f"Provider '{provider}' not configured")
            return await p.generate(prompt, **kwargs)

        # Fallback strategy - use first available
        for name, p in self.providers.items():
            try:
                return await p.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning("Provider {} failed: {}", name, e)
                continue
        raise RuntimeError("No providers available or all failed")

    async def generate_stream(self, prompt: str, provider: Optional[str] = None, **kwargs: Any):
        """Stream generate with real-time token streaming."""
        if provider:
            p = self.providers.get(provider)
            if not p:
                raise ValueError(f"Provider '{provider}' not configured")
            # Check if provider supports streaming
            if hasattr(p, 'generate_stream'):
                async for token in p.generate_stream(prompt, **kwargs):
                    yield token
            else:
                # Fallback to chunked response
                response = await p.generate(prompt, **kwargs)
                for chunk in response.split(" "):
                    yield chunk + " "
        else:
            # Use first available streaming provider
            for name, p in self.providers.items():
                try:
                    if hasattr(p, 'generate_stream'):
                        async for token in p.generate_stream(prompt, **kwargs):
                            yield token
                        return
                    else:
                        response = await p.generate(prompt, **kwargs)
                        for chunk in response.split(" "):
                            yield chunk + " "
                        return
                except Exception as e:
                    logger.warning("Provider {} streaming failed: {}", name, e)
                    continue
            raise RuntimeError("No streaming providers available or all failed")
