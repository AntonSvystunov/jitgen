import asyncio
import time

from jitgen import JitGenError, StreamDriver, create_python_session
from jitgen.segmenters import markdown_code
from openai import AsyncOpenAI

AGENT_PROMPT = (
    "You are a coding agent solving a task that requires actually running "
    "code to get the right answer — do not guess the results yourself. "
    "Respond with a single fenced ```python code block and nothing else. "
    "Write the script as a flat sequence of top-level statements — do not "
    "define any functions or classes, and do not wrap the code in "
    '`if __name__ == "__main__"`. '
    "Given the list of integers "
    "[482, 17, 933, 6, 271, 58, 999, 142, 3, 764, 29, 815, 91, 6003, 47], "
    "write a script that step by step: (1) sorts the list and prints it, "
    "(2) prints the sum, mean, and median, (3) finds and prints every prime "
    "number in the list, and (4) finds and prints the longest strictly "
    "increasing contiguous subsequence. Print each result on its own line "
    "with a `print()` call as soon as it is computed, rather than "
    "collecting everything and printing at the end."
)


def _print_output(output: str) -> None:
    print(
        f"\n\033[36m--- jitgen output ---\n{output}--- end output ---\033[0m\n",
        flush=True,
    )


async def main() -> None:
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="api_key")

    stream = await client.chat.completions.create(
        model="xingyaow/codeact-agent-mistral:latest",
        stream=True,
        messages=[{"role": "user", "content": AGENT_PROMPT}],
    )

    stream_started = time.perf_counter()

    async with create_python_session() as session:
        # markdown_code("python") matches the ```python ... ``` fence the
        # prompt asks the model for; StreamDriver feeds each raw delta
        # through it and into the session, dispatching each top-level
        # statement to the executor as soon as its boundary is fixed rather
        # than waiting for the whole block to finish streaming.
        driver = StreamDriver(session, markdown_code("python"))

        async for event in stream:
            delta = event.choices[0].delta.content
            if delta is None:
                continue
            print(delta, end="", flush=True)

            for output in await driver.apush(delta):
                _print_output(output)

            if driver.has_error:
                # Stop paying for tokens once the model's code has failed.
                break

        try:
            for output in await driver.afinish():
                _print_output(output)
        except JitGenError as exc:
            print(f"\n\033[31m[jitgen error] {exc}\033[0m")

    print()
    stats = session.stats
    time_to_first = stats.since(stream_started)
    if time_to_first is not None:
        print(f"time to first statement output: {time_to_first:.2f}s")
    print(
        f"statements executed: {stats.statements_executed} "
        f"(failed: {stats.statements_failed})"
    )


if __name__ == "__main__":
    asyncio.run(main())
