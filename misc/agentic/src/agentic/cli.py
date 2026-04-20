import argparse
import asyncio
import sys

from dotenv import load_dotenv

import logging

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     stream=sys.stdout,
# )

# Configure telemetry before importing langchain/agents so the SDK reads the
# correct values at initialisation time.
from agentic import telemetry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agentic-eval",
        description="Benchmark incremental vs standard CodeAct agents on DABstep.",
    )
    p.add_argument("--model-name", dest="model_name")
    p.add_argument("--mode", choices=["incremental", "standard"])
    p.add_argument("--dataset", choices=["dev", "full"])
    p.add_argument("--max-steps", dest="max_steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--executor", choices=["sandbox", "inproc"])
    p.add_argument("--results-dir", dest="results_dir")
    p.add_argument("--enable-langsmith", dest="enable_langsmith", action="store_true", default=None)
    p.add_argument("--enable-otel", dest="enable_otel", action="store_true", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Build config from .env + env vars, then overlay any CLI overrides.
    from agentic.config import Config

    # Load environment variables from .env file
    load_dotenv()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg = Config(**overrides)

    # Apply telemetry settings before importing langchain components.
    telemetry.configure(enable_langsmith=cfg.enable_langsmith, enable_otel=cfg.enable_otel)

    from agentic.runner import run_benchmark

    asyncio.run(run_benchmark(cfg))


if __name__ == "__main__":
    main()
