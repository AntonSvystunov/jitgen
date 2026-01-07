import asyncio

from .run import run_evaluation
from dotenv import load_dotenv

_ = load_dotenv()

def main() -> None:
    asyncio.run(run_evaluation())

