# Shared pieces of the PTC examples: the mock `lookup_price` tool, the OpenAI
# client, and the stream-execute-answer loop.
import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, aclosing

from jitgen import JitGenError, Session, StreamDriver
from jitgen_openai import OpenAIToolCallSegmenter
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk

MOCK_PRICES = {
    "AAPL": 178.32,
    "MSFT": 412.65,
    "GOOG": 164.87,
    "AMZN": 186.43,
    "NVDA": 121.79,
}
LOOKUP_DELAY_SECONDS = 0.3

_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


async def lookup_price(symbol: str) -> float:
    """Mock price lookup; the sleep stands in for real network latency.

    Args:
        symbol: The ticker symbol to look up.

    Returns:
        A mock price for `symbol` (a fixed fallback for unknown symbols).
    """
    await asyncio.sleep(LOOKUP_DELAY_SECONDS)
    return MOCK_PRICES.get(symbol.upper(), 100.0)


def build_client() -> AsyncOpenAI:
    """Build an OpenAI client, defaulting to a local LM Studio server.

    LangSmith tracing is a no-op unless `LANGSMITH_TRACING` is set.

    Returns:
        An `AsyncOpenAI` client wrapped for LangSmith tracing.
    """
    return wrap_openai(
        AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
        )
    )


def require_model() -> str:
    """Read the model identifier from `OPENAI_MODEL`.

    Returns:
        The configured model identifier.

    Raises:
        RuntimeError: If `OPENAI_MODEL` is unset or empty.
    """
    model = os.environ.get("OPENAI_MODEL", "")
    if not model:
        msg = "set OPENAI_MODEL to the identifier of a tool-calling model"
        raise RuntimeError(msg)
    return model


def eval_tool(description: str, code_description: str) -> dict[str, object]:
    """Build the `eval` tool schema, whose single `code` argument is source code.

    Args:
        description: What the tool does, shown to the model.
        code_description: How to write the `code` argument, shown to the model.

    Returns:
        An OpenAI function-tool definition.
    """
    return {
        "type": "function",
        "function": {
            "name": "eval",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": code_description}
                },
                "required": ["code"],
            },
        },
    }


async def create_eval_stream(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, object]],
    tool: dict[str, object],
) -> AsyncStream[ChatCompletionChunk]:
    """Start a streamed turn in which the model must call `tool`.

    Args:
        client: The OpenAI-compatible client to stream from.
        model: The model identifier to request.
        messages: The conversation so far.
        tool: The `eval` tool definition.

    Returns:
        The chunk stream; use it as an async context manager so it is closed
        even when iteration stops early.
    """
    return await client.chat.completions.create(
        model=model,
        stream=True,
        messages=messages,
        tools=[tool],
        # Some local servers (e.g. LM Studio) reject the object form of
        # `tool_choice`; with a single tool, "required" is equivalent.
        tool_choice="required",
    )


async def iter_argument_deltas(
    stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncGenerator[tuple[str | None, str]]:
    """Yield the first tool call's `arguments` deltas as they arrive.

    Args:
        stream: A streamed chat completion.

    Yields:
        `(tool_call_id, arguments_delta)`; the id is `None` on chunks that
        don't carry it.
    """
    async for event in stream:
        if not event.choices:
            continue
        for call in event.choices[0].delta.tool_calls or ():
            if call.index != 0:
                continue
            arguments = call.function.arguments if call.function else None
            if call.id or arguments:
                yield call.id, arguments or ""


def _print_output(output: str) -> None:
    print(f"\n{_CYAN}--- jitgen output ---\n{output}--- end output ---{_RESET}\n")


async def _stream_eval_call(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, object]],
    tool: dict[str, object],
    session: Session,
) -> tuple[str, str, str]:
    """Stream one turn, executing the `eval` tool call's code as it arrives.

    Args:
        client: The OpenAI-compatible client to stream from.
        model: The model identifier to request.
        messages: The conversation so far.
        tool: The `eval` tool definition.
        session: The session that executes the streamed code.

    Returns:
        `(tool_call_id, raw_arguments, tool_result)`, where `tool_result` is
        the captured output or a description of the execution error.
    """
    driver = StreamDriver(session, OpenAIToolCallSegmenter(property_name="code"))
    tool_call_id = ""
    raw_arguments: list[str] = []
    captured: list[str] = []

    stream = await create_eval_stream(client, model, messages, tool)
    async with stream, aclosing(iter_argument_deltas(stream)) as argument_deltas:
        async for call_id, arguments_delta in argument_deltas:
            tool_call_id = call_id or tool_call_id
            if not arguments_delta:
                continue
            raw_arguments.append(arguments_delta)
            print(arguments_delta, end="", flush=True)
            for output in await driver.apush(arguments_delta):
                captured.append(output)
                _print_output(output)
            if driver.has_error:
                break

    try:
        for output in await driver.afinish():
            captured.append(output)
            _print_output(output)
        tool_result = "".join(captured) or "(no output)"
    except JitGenError as exc:
        tool_result = f"execution error: {exc}"
        print(f"\n{_RED}[jitgen error] {exc}{_RESET}")

    if len(raw_arguments) < 2:
        print(
            f"\n{_YELLOW}[warning] the `code` argument arrived in a single "
            "chunk, so incremental execution wasn't exercised this run."
            f"{_RESET}"
        )
    return tool_call_id, "".join(raw_arguments), tool_result


async def _stream_answer(
    client: AsyncOpenAI, model: str, messages: list[dict[str, object]]
) -> str:
    """Stream and print the model's plain-text answer.

    Args:
        client: The OpenAI-compatible client to stream from.
        model: The model identifier to request.
        messages: The conversation so far.

    Returns:
        The full answer text.
    """
    print(f"\n{_MAGENTA}--- final answer ---{_RESET}")
    answer: list[str] = []
    stream = await client.chat.completions.create(
        model=model, stream=True, messages=messages
    )
    async with stream:
        async for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                answer.append(delta)
                print(delta, end="", flush=True)
    print()
    return "".join(answer)


async def run(
    task: str,
    *,
    system_prompt: str,
    tool: dict[str, object],
    open_session: Callable[[], AbstractAsyncContextManager[Session]],
) -> str:
    """Solve `task` via one PTC round trip: stream `eval`, execute, answer.

    Args:
        task: The user's task.
        system_prompt: The system message describing the `eval` tool.
        tool: The `eval` tool definition.
        open_session: Opens the session that executes the tool call's code.

    Returns:
        The model's final answer.
    """
    client = build_client()
    model = require_model()
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    async with open_session() as session:
        tool_call_id, raw_arguments, tool_result = await _stream_eval_call(
            client, model, messages, tool, session
        )
    if not tool_call_id:
        return "(model did not call the eval tool)"

    messages += [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "eval", "arguments": raw_arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": tool_call_id, "content": tool_result},
    ]
    return await _stream_answer(client, model, messages)
