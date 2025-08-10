from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Awaitable
from loguru import logger

from ..providers.manager import AIProviderManager
from ..tools.manager import ToolManager
from ..mcp.connector import MCPConnector
from ..config.settings import EnergoConfig
from .policy import PolicyEnforcer


@dataclass
class AgentProfile:
    name: str = "Energo"
    personality: str = (
        "Enthusiastic, energetic, helpful, friendly, and smart (more than intelligent)."
    )
    description: str = (
        "Energo is a super agent that can connect to MCP servers, call tools, interpret code, "
        "and read the file system. Designed for flexibility, privacy, and accountability."
    )


class EnergoAgent:
    def __init__(
        self,
        config: EnergoConfig,
        provider_manager: Optional[AIProviderManager] = None,
        tool_manager: Optional[ToolManager] = None,
        mcp_connector: Optional[MCPConnector] = None,
        profile: Optional[AgentProfile] = None,
    ) -> None:
        self.config = config
        self.profile = profile or AgentProfile()
        self.providers = provider_manager or AIProviderManager(config)
        self.tools = tool_manager or ToolManager(config)
        self.mcp = mcp_connector or MCPConnector(config)
        self.policy = PolicyEnforcer()

        logger.debug("EnergoAgent initialized with profile: {}", self.profile)

    async def initialize(self) -> None:
        await asyncio.gather(
            self.providers.initialize(),
            self.tools.initialize(),
            self.mcp.initialize(),
        )
        logger.info("Energo initialized and ready!")

    async def ask(self, prompt: str, provider: Optional[str] = None, **kwargs: Any) -> str:
        """Route a prompt to the best provider or a specified one."""
        logger.debug("Received prompt: {}", prompt)
        # Policy check
        record = self.policy.pre_action("ask", metadata={"provider": provider})
        self.policy.audit(record)
        response = await self.providers.generate(prompt, provider=provider, **kwargs)
        logger.debug("Provider response: {}", self.policy.redact(response))
        return response

    async def ask_stream(self, prompt: str, provider: Optional[str] = None, **kwargs: Any):
        """Stream tokens in real-time from provider."""
        logger.debug("Received prompt for streaming: {}", prompt)
        record = self.policy.pre_action("ask_stream", metadata={"provider": provider})
        self.policy.audit(record)
        async for token in self.providers.generate_stream(prompt, provider=provider, **kwargs):
            yield token

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        logger.debug("Calling tool: {} with args {}", name, kwargs)
        record = self.policy.pre_action("call_tool", target=name, metadata=kwargs)
        self.policy.audit(record)
        result = await self.tools.call(name, **kwargs)
        logger.debug("Tool result: {}", self.policy.redact(result))
        return result

    async def mcp_call(self, server: str, method: str, **kwargs: Any) -> Any:
        logger.debug("MCP call: server={} method={} args={}", server, method, kwargs)
        record = self.policy.pre_action("mcp_call", target=f"{server}.{method}", metadata=kwargs)
        self.policy.audit(record)
        result = await self.mcp.call(server, method, **kwargs)
        logger.debug("MCP result: {}", self.policy.redact(result))
        return result

    async def interpret_code(self, code: str, language: Optional[str] = None) -> Any:
        """Very basic code interpretation; can be extended to use sandboxes."""
        logger.debug("Interpreting code in language {}", language)
        # For now, just route to providers with system prompt
        system_prompt = (
            "You are Energo, a highly capable code interpreter. Analyze the given code and respond with "
            "insights, potential issues, and execution plan. Do not execute untrusted code."
        )
        prompt = f"{system_prompt}\n\nLanguage: {language or 'auto'}\n\nCode:\n{code}"
        return await self.ask(prompt)

    async def read_file(self, path: str) -> str:
        logger.debug("Reading file: {}", path)
        return await self.tools.call("read_file", path=path)

    async def run(self):
        await self.initialize()
        logger.info("Energo is running. No UI mode.")
        # Placeholder main loop or API could be added here
        # For now, just idle
        while True:
            await asyncio.sleep(3600)
