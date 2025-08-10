from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Callable, Dict
from loguru import logger
from ..config.settings import EnergoConfig


ToolFunc = Callable[..., Awaitable[Any]]


class ToolManager:
    def __init__(self, config: EnergoConfig) -> None:
        self.config = config
        self.tools: Dict[str, ToolFunc] = {}

    async def initialize(self) -> None:
        logger.info("Initializing tools ...")
        # Register built-in tools
        self.register("read_file", self.read_file)
        self.register("list_dir", self.list_dir)
        logger.info("Tools ready: {}", list(self.tools.keys()))

    def register(self, name: str, func: ToolFunc) -> None:
        self.tools[name] = func

    async def call(self, name: str, **kwargs: Any) -> Any:
        fn = self.tools.get(name)
        if not fn:
            raise ValueError(f"Tool '{name}' not found")
        return await fn(**kwargs)

    # Built-in tool implementations
    async def read_file(self, path: str) -> str:
        logger.debug("Tool.read_file path={}", path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    async def list_dir(self, path: str) -> Any:
        logger.debug("Tool.list_dir path={}", path)
        import os
        return [
            {"name": name, "is_dir": os.path.isdir(os.path.join(path, name))}
            for name in os.listdir(path)
        ]
