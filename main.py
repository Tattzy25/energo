import asyncio
from loguru import logger
from .config.settings import EnergoConfig
from .core.agent import EnergoAgent


async def main():
    logger.add(lambda msg: print(msg, end=""))
    cfg = EnergoConfig()
    agent = EnergoAgent(cfg)
    await agent.initialize()

    # Quick sanity: ask using first available provider if configured
    try:
        resp = await agent.ask("Say hi as Energo and confirm core is up.")
        print("\nEnergo:", resp)
    except Exception as e:
        print("No provider configured or request failed:", e)

    # Idle (no UI)
    # await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
