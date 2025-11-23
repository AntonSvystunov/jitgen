import asyncio

from .run import run_evaluation

def main() -> None:
    asyncio.run(run_evaluation())

