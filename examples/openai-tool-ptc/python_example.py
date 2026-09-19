# Programmatic Tool Calling (PTC): the model is given one tool, `eval`,
# whose `code` argument is Python source. Instead of waiting for the tool
# call's `arguments` to finish streaming before running anything, jitgen
# executes each top-level statement of `code` as soon as its boundary is
# provably fixed — while the rest of the JSON object (and any properties
# after `code`) is still arriving. The captured stdout is then sent back to
# the model as the tool result so it can answer from the real, computed
# value rather than guessing it. See docs/METHOD.md for the underlying
# method, and https://docs.langchain.com/oss/python/deepagents/interpreters
# for the `eval`-tool convention this borrows.
import asyncio
import os

from dotenv import load_dotenv
from jitgen import (
    InProcPythonExecutor,
    JitGenError,
    StreamDriver,
    create_python_session,
)
from jitgen_openai import OpenAIToolCallSegmenter
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    "You solve tasks by writing Python code and calling the `eval` tool to "
    "run it — never compute or guess a numeric result yourself. Inside "
    "that code, an `async def lookup_price(symbol: str) -> float` "
    "function is already available (a Programmatic Tool Calling helper — "
    "do not define or import it, just call it) that returns the current "
    "price of a ticker symbol. It is a coroutine function, so call it as "
    "`await lookup_price(symbol)` — top-level `await` is allowed directly "
    "in this code, do not wrap it in `async def main(): ...` or "
    "`asyncio.run(...)`. Call it once per symbol and print the result "
    "immediately after each call, before moving on to the next symbol. "
    "Write the code as a flat sequence of top-level statements (no "
    "function/class definitions of your own), and print every value you "
    "need with print() as soon as it is computed."
)

USER_TASK = (
    "Look up the current price of AAPL, MSFT, GOOG, AMZN, and NVDA with "
    "`lookup_price`, then compute and print the total cost of buying 10 "
    "shares of each."
)

EVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "eval",
        "description": (
            "Execute Python code and return what it prints. Use this for "
            "any computation instead of answering from memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source code to execute. An "
                        "`async def lookup_price(symbol: str) -> float` "
                        "function is already available in this "
                        "environment — call it with `await "
                        "lookup_price(...)`, top-level `await` works "
                        "directly here. It needs no import or definition."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

_MOCK_PRICES = {
    "AAPL": 178.32,
    "MSFT": 412.65,
    "GOOG": 164.87,
    "AMZN": 186.43,
    "NVDA": 121.79,
}
_LOOKUP_DELAY_SECONDS = 0.3


async def _lookup_price(symbol: str) -> float:
    """Programmatic Tool Calling helper: a mock live price lookup.

    Async because a real version of this would be an `await`ed network
    call (a price API, a broker SDK); `asyncio.sleep` stands in for that
    latency without blocking the thread it runs on. Registered with
    `InProcPythonExecutor(tools=...)` and called directly from the
    executed code as `await lookup_price(...)` — `InProcPythonExecutor`
    compiles each dispatched statement with `ast.PyCF_ALLOW_TOP_LEVEL_
    AWAIT`, so a coroutine call like this one runs to completion on the
    executor's worker thread without needing a synchronous bridge.

    Args:
        symbol: The ticker symbol to look up.

    Returns:
        A mock price for `symbol` (a fixed fallback for unknown symbols).
    """
    await asyncio.sleep(_LOOKUP_DELAY_SECONDS)
    return _MOCK_PRICES.get(symbol.upper(), 100.0)


def _build_client() -> AsyncOpenAI:
    """Build an OpenAI client pointed at a local LM Studio server.

    Point `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL` at any other
    OpenAI-compatible endpoint to reuse this example unchanged, as long as
    it supports streaming tool-call arguments.

    Wrapped with `wrap_openai` for LangSmith tracing, same as the providers
    in `misc/mbpp`. Tracing is opt-in: it only sends anything when the
    standard LangSmith env vars (`LANGSMITH_TRACING`/`LANGSMITH_API_KEY`)
    are set; otherwise the wrapper is a no-op passthrough.

    Returns:
        An `AsyncOpenAI` client configured for a local LM Studio server.
    """
    return wrap_openai(
        AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
        )
    )


def _print_output(output: str) -> None:
    print(
        f"\n\033[36m--- jitgen output ---\n{output}--- end output ---\033[0m\n",
        flush=True,
    )


async def _stream_eval_call(
    client: AsyncOpenAI, model: str, messages: list[dict[str, object]]
) -> tuple[str, str, str]:
    """Stream one turn, executing the `eval` tool call's code as it arrives.

    Args:
        client: The OpenAI-compatible client to stream from.
        model: The model identifier to request.
        messages: The conversation so far.

    Returns:
        `(tool_call_id, raw_arguments, tool_result)` — the id and raw
        (still-JSON-encoded) arguments needed to reconstruct the
        assistant's tool-call message, and the text to send back as the
        tool result (the captured stdout, or a description of the
        execution error).
    """
    stream = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=messages,
        tools=[EVAL_TOOL],
        # "required" rather than forcing {"type": "function", "function": {...}}
        # — several OpenAI-compatible local servers (e.g. LM Studio's
        # llama.cpp backend) reject the object form of `tool_choice` and only
        # support the string values. With one tool offered, "required" is
        # just as deterministic here.
        tool_choice="required",
    )

    tool_call_id = ""
    raw_arguments_parts: list[str] = []
    argument_delta_count = 0
    captured_stdout = ""
    tool_result = ""

    executor = InProcPythonExecutor(tools={"lookup_price": _lookup_price})
    async with executor, create_python_session(executor=executor) as session:
        driver = StreamDriver(session, OpenAIToolCallSegmenter(property_name="code"))

        async for event in stream:
            tool_calls = event.choices[0].delta.tool_calls
            if not tool_calls:
                continue
            call = tool_calls[0]
            if call.id:
                tool_call_id = call.id
            if call.function is None or call.function.arguments is None:
                continue

            arguments_delta = call.function.arguments
            raw_arguments_parts.append(arguments_delta)
            argument_delta_count += 1
            print(arguments_delta, end="", flush=True)

            for output in await driver.apush(arguments_delta):
                captured_stdout += output
                _print_output(output)

            if driver.has_error:
                break

        try:
            for output in await driver.afinish():
                captured_stdout += output
                _print_output(output)
            tool_result = captured_stdout or "(no output)"
        except JitGenError as exc:
            tool_result = f"execution error: {exc}"
            print(f"\n\033[31m[jitgen error] {exc}\033[0m")

    if argument_delta_count < 2:
        print(
            "\n\033[33m[warning] the `code` argument arrived in a single "
            "chunk — this model/runtime buffered the whole tool call "
            "instead of streaming its arguments token-by-token, so jitgen's "
            "incremental-execution benefit wasn't actually exercised this "
            "run.\033[0m"
        )

    return tool_call_id, "".join(raw_arguments_parts), tool_result


async def run(task: str) -> str:
    """Solve `task` via one PTC round trip: stream `eval`, execute, answer.

    Args:
        task: The user's task.

    Returns:
        The model's final natural-language answer.
    """
    client = _build_client()
    model = os.environ.get("OPENAI_MODEL", "")
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    tool_call_id, raw_arguments, tool_result = await _stream_eval_call(
        client, model, messages
    )
    if not tool_call_id:
        return "(model did not call the eval tool)"

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "eval", "arguments": raw_arguments},
                }
            ],
        }
    )
    messages.append(
        {"role": "tool", "tool_call_id": tool_call_id, "content": tool_result}
    )

    print("\n\033[35m--- final answer ---\033[0m")
    final_stream = await client.chat.completions.create(
        model=model, stream=True, messages=messages
    )
    answer = ""
    async for event in final_stream:
        delta = event.choices[0].delta.content
        if delta:
            answer += delta
            print(delta, end="", flush=True)
    print()
    return answer


async def main() -> None:
    load_dotenv()

    await run(USER_TASK)


if __name__ == "__main__":
    asyncio.run(main())
