# Runs the fetch task through `PtcAgent` (execute after the tool call has
# streamed) and then `IptcAgent` (execute while it streams), and prints each
# wall time. Both use the same model seed, so they tend to write the same code;
# with tracing on, the two runs sit side by side in LangSmith.
import argparse
import asyncio
import os
import time

from jitgen_openai import IptcAgent, McpToolBridge, PtcAgent

from agent_example import FETCH_SERVER, SYSTEM_PROMPTS, USER_TASK
from utils import add_language_argument, run_agent

SEED = int(os.environ.get("COMPARE_SEED", "42"))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PtcAgent and IptcAgent.")
    add_language_argument(parser)
    args = parser.parse_args()

    timings: dict[str, float] = {}
    for agent_cls in (PtcAgent, IptcAgent):
        print(f"\n===== {agent_cls.__name__} ({args.language}) =====")
        start = time.monotonic()
        await run_agent(
            McpToolBridge.stdio(FETCH_SERVER),
            USER_TASK,
            system_prompt=SYSTEM_PROMPTS[args.language],
            agent_cls=agent_cls,
            language=args.language,
            completion_options={"seed": SEED},
        )
        timings[agent_cls.__name__] = time.monotonic() - start

    print(f"\n===== wall time ({args.language}, seed {SEED}) =====")
    for name, seconds in timings.items():
        print(f"{name:10} {seconds:.2f}s")
    print("A seed makes sampling repeatable, not guaranteed: check both runs wrote")
    print("the same code, and compare over several seeds.")


if __name__ == "__main__":
    asyncio.run(main())
