import asyncio
import logging
import sys
from tqdm import tqdm

from .run import run_evaluation
from dotenv import load_dotenv

_ = load_dotenv()

logger = logging.getLogger(__name__)

def main() -> None:
    try:
        asyncio.run(run_evaluation())
    except KeyboardInterrupt:
        tqdm.write("\n⚠️  Evaluation interrupted by user", file=sys.stderr)
    except Exception as e:
        tqdm.write(f"\n❌ Fatal error: {e}", file=sys.stderr)
        logger.exception(f"Fatal error during evaluation: {e}")
        raise

