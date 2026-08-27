# Streams a coding agent's response through jitgen so each top-level
# statement in its ```python code block runs as soon as the model finishes
# writing it, instead of waiting for the whole block to finish streaming.
# Tracing is optional: set LANGCHAIN_TRACING=true (see .env) to send a
# LangSmith trace of the run, including a "dispatch"/"flush" span for every
# jitgen call that actually produced output.
import asyncio
import os

from dotenv import load_dotenv
from jitgen import JitGenError, StreamDriver, create_python_session
from jitgen.segmenters import markdown_code
from langsmith import get_current_run_tree, traceable
from langsmith.wrappers import wrap_openai
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


def _build_client() -> AsyncOpenAI:
    """Build an OpenAI client wrapped for LangSmith tracing.

    Defaults to a local Ollama server so the example runs with no API key.
    Point `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL` at any other
    OpenAI-compatible endpoint to reuse this example unchanged.

    Returns:
        An `AsyncOpenAI` client whose calls are recorded as LangSmith runs.
    """
    return wrap_openai(
        AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        )
    )


def _print_output(output: str) -> None:
    print(
        f"\n\033[36m--- jitgen output ---\n{output}--- end output ---\033[0m\n",
        flush=True,
    )


def _log_tool_span(name: str, inputs: dict[str, str], outputs: dict[str, str]) -> None:
    """Record a LangSmith tool span for a jitgen dispatch/flush call.

    Only called once there's output worth recording: most dispatch calls
    return nothing because a statement boundary hasn't closed yet, and
    tracing every one of them would swamp the trace with empty, sub-
    millisecond spans.

    Args:
        name: Span name, `"dispatch"` or `"flush"`.
        inputs: Run inputs to record, e.g. the raw model delta.
        outputs: Run outputs to record, e.g. the stdout it produced.
    """
    parent = get_current_run_tree()
    if parent is None:
        return

    child = parent.create_child(name=name, run_type="tool", inputs=inputs)
    child.end(outputs=outputs)
    child.post()


@traceable(run_type="chain", name="JitGen")
async def run(prompt: str) -> str:
    """Stream a model's response through jitgen and return its stdout.

    Args:
        prompt: The prompt sent to the model.

    Returns:
        The concatenated stdout produced while executing the model's code.
    """
    client = _build_client()
    stream = await client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "xingyaow/codeact-agent-mistral:latest"),
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )

    response = ""

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

            stdout = ""
            for output in await driver.apush(delta):
                stdout += output
                _print_output(output)
            if stdout:
                _log_tool_span("dispatch", {"delta": delta}, {"stdout": stdout})
                response += stdout

            if driver.has_error:
                # Stop paying for tokens once the model's code has failed.
                break

        try:
            stdout = ""
            for output in await driver.afinish():
                stdout += output
                _print_output(output)
            if stdout:
                _log_tool_span("flush", {}, {"stdout": stdout})
                response += stdout
        except JitGenError as exc:
            print(f"\n\033[31m[jitgen error] {exc}\033[0m")

    return response


async def main() -> None:
    load_dotenv()

    await run(AGENT_PROMPT)


if __name__ == "__main__":
    asyncio.run(main())
