from __future__ import annotations
import abc
from typing import Any, Dict
from loguru import logger
import os

from ..config.settings import EnergoConfig


class BaseProvider(abc.ABC):
    def __init__(self, config: EnergoConfig) -> None:
        self.config = config

    @abc.abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


class OpenAIProvider(BaseProvider):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        from openai import AsyncOpenAI
        # If AI Gateway is configured, route OpenAI calls through it
        base_url = self.config.ai_gateway_base_url if self.config.ai_gateway_api_key else None
        api_key = self.config.ai_gateway_api_key or self.config.openai_api_key
        client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
        model = kwargs.get("model", self.config.openai_model)
        system = kwargs.get("system", "You are Energo, an enthusiastic and helpful assistant.")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.3),
        )
        return response.choices[0].message.content

    async def generate_stream(self, prompt: str, **kwargs: Any):
        """Streaming support via OpenAI SDK; works with AI Gateway when base_url set."""
        from openai import AsyncOpenAI
        base_url = self.config.ai_gateway_base_url if self.config.ai_gateway_api_key else None
        api_key = self.config.ai_gateway_api_key or self.config.openai_api_key
        client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
        model = kwargs.get("model", self.config.openai_model)
        system = kwargs.get("system", "You are Energo, an enthusiastic and helpful assistant.")
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.3),
            stream=True,
        )
        async for event in stream:
            # OpenAI SDK v1 streaming returns chunks with choices[0].delta
            delta = event.choices[0].delta.content if getattr(event.choices[0], "delta", None) else None
            if delta:
                yield delta


class AnthropicProvider(BaseProvider):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)
        model = kwargs.get("model", self.config.anthropic_model)
        system = kwargs.get("system", "You are Energo, an enthusiastic and helpful assistant.")
        msg = await client.messages.create(
            model=model,
            system=system,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.3),
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic SDK returns a list of content blocks
        chunks = [c.text for c in msg.content if getattr(c, "type", None) == "text"]
        return "".join(chunks)


class GroqProvider(BaseProvider):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=self.config.groq_api_key)
        model = kwargs.get("model", self.config.groq_model)
        system = kwargs.get("system", "You are Energo, an enthusiastic and helpful assistant.")
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.3),
        )
        return completion.choices[0].message.content

    async def generate_stream(self, prompt: str, **kwargs: Any):
        """Yield tokens as they arrive from Groq chat.completions (true streaming)."""
        from groq import AsyncGroq
        client = AsyncGroq(api_key=self.config.groq_api_key)
        model = kwargs.get("model", self.config.groq_model)
        system = kwargs.get("system", "You are Energo, an enthusiastic and helpful assistant.")
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.3),
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta.content if getattr(event.choices[0], "delta", None) else None
            if delta:
                yield delta


class GoogleVertexProvider(BaseProvider):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        # Placeholder for Vertex AI (Gemini) chat implementation
        # Avoid heavy setup; return a simple echo for now or raise if not configured
        logger.warning("Google Vertex provider not fully implemented yet.")
        return "[Google Vertex placeholder] " + prompt[:200]
