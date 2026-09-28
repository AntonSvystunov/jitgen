# Agent over PARCS, a distributed C# compute cluster exposed over MCP (legacy
# HTTP+SSE). The model writes one flat Python (or, with `--language
# javascript`, JavaScript) `eval` call that sizes the job, compiles C# on the
# cluster, runs a parallel layer and an aggregation layer.
import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from jitgen_openai import CodeLanguage, IptcAgent, McpToolBridge, PtcAgent

from utils import add_language_argument, run_agent

PROMPT_FILES: dict[CodeLanguage, str] = {
    "python": "parcs_prompt.md",
    "javascript": "parcs_prompt_js.md",
}

TASKS = {
    "primes": (
        "Using the PARCS cluster, count the prime numbers p with "
        "1,000,000,000 <= p < 1,010,000,000. In a first layer, split the range "
        "across workers so each one sieves its own sub-range; then add up the "
        "per-worker counts in a single-worker aggregation layer. Report the "
        "count and how many workers did the sieving."
    ),
    "var": (
        "Using the PARCS cluster, estimate the 1-day 99% Value-at-Risk and "
        "Conditional VaR of an equally weighted portfolio of 50 assets. Inside "
        "the C# code, build the return covariance matrix as "
        "Sigma = A^T A / 50 + 0.01 I from a 50x50 matrix A of standard normal "
        "draws with a fixed seed, and take its Cholesky factor. Simulate "
        "2,000,000 scenarios split evenly across workers, each worker with its "
        "own seed; loss = -(portfolio return). Aggregate in a single-worker "
        "final layer and report var_99 and cvar_99."
    ),
}
# Computed independently with a segmented sieve.
EXPECTED = {"primes": "482449"}
AGENTS = {"iptc": IptcAgent, "ptc": PtcAgent}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent on the PARCS cluster.")
    parser.add_argument(
        "--task",
        default="primes",
        help=f"a preset ({', '.join(TASKS)}) or the task text itself",
    )
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        default="iptc",
        help="iptc executes while the code streams; ptc waits for the whole call",
    )
    parser.add_argument("--seed", type=int, help="model sampling seed")
    add_language_argument(parser)
    return parser.parse_args()


def _parcs_url() -> str:
    """Read the PARCS MCP endpoint, which is kept out of the repository.

    Returns:
        The SSE endpoint URL from `PARCS_SERVER_URL`.

    Raises:
        SystemExit: If `PARCS_SERVER_URL` is unset.
    """
    load_dotenv()
    url = os.environ.get("PARCS_SERVER_URL", "")
    if not url:
        msg = "set PARCS_SERVER_URL (e.g. in .env) to the PARCS MCP SSE endpoint"
        raise SystemExit(msg)
    return url


async def main() -> None:
    args = _parse_args()
    prompt_path = Path(__file__).parent / PROMPT_FILES[args.language]
    answer = await run_agent(
        McpToolBridge.sse(_parcs_url()),
        TASKS.get(args.task, args.task),
        system_prompt=prompt_path.read_text(encoding="utf-8"),
        agent_cls=AGENTS[args.agent],
        language=args.language,
        # Starting worker pods on a cold cluster can take minutes, and each
        # retry after a compile error or failed layer costs a turn.
        max_turns=16,
        executor_timeout=600.0,
        completion_options=None if args.seed is None else {"seed": args.seed},
    )
    if args.task in EXPECTED:
        found = EXPECTED[args.task] in answer.replace(",", "")
        print(f"\nExpected {EXPECTED[args.task]}: {'found' if found else 'NOT found'}")


if __name__ == "__main__":
    asyncio.run(main())
