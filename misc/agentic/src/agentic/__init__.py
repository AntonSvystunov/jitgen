from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from .config import load_config
from .run import run_evaluation

_ = load_dotenv(override=True)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
	config = load_config(argv)

	try:
		asyncio.run(run_evaluation(config))
	except KeyboardInterrupt:
		tqdm.write("\nEvaluation interrupted by user", file=sys.stderr)
	except Exception as error:
		tqdm.write(f"\nFatal error: {error}", file=sys.stderr)
		logger.exception("Fatal error during evaluation")
		raise
