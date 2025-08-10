from __future__ import annotations
from typing import Any, AsyncGenerator, Optional
from loguru import logger
from ..config.settings import EnergoConfig


class GroqStreaming:
    def __init__(self, config: EnergoConfig) -> None:
        self.config = config

    async def chat_stream(self, prompt: str, *, model: Optional[str] = None, system: Optional[str] = None, temperature: float = 0.3) -> AsyncGenerator[str, None]:
        # True server-sent streaming using Groq's async SDK
        from groq import AsyncGroq
        client = AsyncGroq(api_key=self.config.groq_api_key)
        mdl = model or self.config.groq_model
        sys = system or "You are Energo, an enthusiastic and helpful assistant."

        stream = await client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta.content if getattr(event.choices[0], "delta", None) else None
            if delta:
                yield delta
