# Adapted from misc/iptc-parcs/iptc_parcs/instrumentation.py; chunk times are
# taken when a chunk arrives rather than when the agent reads it.
import asyncio
import contextlib
import json
from typing import Any, Self

import tiktoken

from iptc_bfcl.metrics import RunRecorder, TurnRecord

_ENCODING = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Proxy token count (tiktoken `o200k_base`), identical for every strategy."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def estimate_input_tokens(messages: list[Any], tools: list[Any] | None) -> int:
    """Proxy count of the prompt: every message plus the tool schemas."""
    text = json.dumps(messages, default=str, ensure_ascii=False)
    if tools:
        text += json.dumps(tools, default=str, ensure_ascii=False)
    return count_tokens(text)


def truncate_tool_results(
    messages: list[Any], limit: int | None
) -> tuple[list[Any], int]:
    """Cap every `tool` message's content at `limit` characters.

    The same harness policy for every strategy: native tool replies
    (baseline) and `eval` output (PTC/IPTC) alike. The agent's own message
    list isn't modified; the cap is re-applied to each request.

    Args:
        messages: The request's messages.
        limit: Maximum characters per tool message, or `None` for no cap.

    Returns:
        The messages to send, and how many were truncated.
    """
    if limit is None:
        return messages, 0
    capped, truncated = [], 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if (
            isinstance(message, dict)
            and message.get("role") == "tool"
            and isinstance(content, str)
            and len(content) > limit
        ):
            note = f"\n[truncated: {len(content) - limit:,} of {len(content):,} characters omitted]"
            message = {**message, "content": content[:limit] + note}
            truncated += 1
        capped.append(message)
    return capped, truncated


def _delta_reasoning(delta: Any) -> str | None:
    for name in ("reasoning_content", "reasoning"):
        value = getattr(delta, name, None)
        if value is None and getattr(delta, "model_extra", None):
            value = delta.model_extra.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _chunk_provider(chunk: Any) -> str | None:
    """The upstream provider OpenRouter routed the call to, if it says."""
    provider = getattr(chunk, "provider", None)
    if provider is None and getattr(chunk, "model_extra", None):
        provider = chunk.model_extra.get("provider")
    return provider if isinstance(provider, str) else None


# Queued after the last chunk, or with the exception that ended the stream.
_END = object()


class _InstrumentedStream:
    """Passes chunks through unchanged while recording one turn.

    A reader task drains the model stream as it arrives and stamps each chunk
    with its arrival time. Timings come from those stamps: IPTC stops reading
    while a closed `code` block finishes executing, so a chunk's read time can
    be long after the model generated it. Text and token estimates are still
    recorded as the agent consumes chunks, so chunks still queued when the
    agent closes the stream early count as never received.
    """

    def __init__(self, inner: Any, recorder: RunRecorder, turn: TurnRecord) -> None:
        self._inner = inner
        self._recorder = recorder
        self._turn = turn
        self._iterator: Any = None
        self._queue: asyncio.Queue[tuple[float, Any, BaseException | None]] | None = (
            None
        )
        self._reader: asyncio.Task[None] | None = None
        self._finished = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._finish()
        await self._stop_reader()
        await self._close_iterator()
        exit_ = getattr(self._inner, "__aexit__", None)
        if exit_ is not None:
            await exit_(*exc_info)
        else:
            await self._close_inner()

    def __aiter__(self) -> "_InstrumentedStream":
        return self

    async def __anext__(self) -> Any:
        if self._queue is None:
            self._iterator = self._inner.__aiter__()
            self._queue = asyncio.Queue()
            self._reader = asyncio.create_task(self._read(self._queue))
        arrived, chunk, error = await self._queue.get()
        if error is not None:
            raise error
        if chunk is _END:
            self._turn.completed = True
            self._finish()
            raise StopAsyncIteration
        self._record(chunk, arrived)
        return chunk

    async def _read(
        self, queue: asyncio.Queue[tuple[float, Any, BaseException | None]]
    ) -> None:
        """Move chunks from the model stream to `queue`, stamped on arrival."""
        try:
            while True:
                chunk = await anext(self._iterator)
                queue.put_nowait((self._recorder.now(), chunk, None))
        except StopAsyncIteration:
            queue.put_nowait((self._recorder.now(), _END, None))
        except Exception as exc:  # noqa: BLE001  # re-raised to the consumer
            queue.put_nowait((self._recorder.now(), _END, exc))

    async def _stop_reader(self) -> None:
        """Cancel the reader before closing the stream it is iterating."""
        reader, self._reader = self._reader, None
        if reader is None:
            return
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader

    async def close(self) -> None:
        self._finish()
        await self._stop_reader()
        await self._close_iterator()
        await self._close_inner()

    async def _close_iterator(self) -> None:
        aclose = getattr(self._iterator, "aclose", None)
        if aclose is not None:
            await aclose()

    async def _close_inner(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            await close()

    def _record(self, chunk: Any, now: float) -> None:
        turn = self._turn
        turn.chunks += 1
        if turn.first_chunk is None:
            turn.first_chunk = now
            turn.generation_id = getattr(chunk, "id", None)
            turn.upstream_provider = _chunk_provider(chunk)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            turn.usage_input = usage.prompt_tokens
            turn.usage_output = usage.completion_tokens
            details = getattr(usage, "completion_tokens_details", None)
            turn.usage_reasoning = getattr(details, "reasoning_tokens", None)
        if not chunk.choices:
            return
        choice = chunk.choices[0]
        if choice.finish_reason:
            turn.finish_reason = choice.finish_reason
        delta = choice.delta
        reasoning = _delta_reasoning(delta)
        if reasoning:
            turn.first_reasoning = turn.first_reasoning or now
            turn.last_token = now
            turn.reasoning_text += reasoning
        if delta.content:
            turn.first_content = turn.first_content or now
            turn.last_token = now
            turn.content_text += delta.content
        for call in delta.tool_calls or ():
            arguments = call.function.arguments if call.function else None
            if arguments:
                turn.first_arguments = turn.first_arguments or now
                turn.last_arguments = turn.last_token = now
                turn.arguments_text += arguments

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        turn = self._turn
        turn.end = self._recorder.now()
        turn.est_reasoning = count_tokens(turn.reasoning_text)
        turn.est_output = turn.est_reasoning + count_tokens(
            turn.content_text + turn.arguments_text
        )


class _Completions:
    def __init__(
        self,
        inner: Any,
        recorder: RunRecorder,
        extra: dict[str, Any],
        tool_result_limit: int | None,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._extra = extra
        self._tool_result_limit = tool_result_limit

    async def create(self, **kwargs: Any) -> _InstrumentedStream:
        kwargs = {**self._extra, **kwargs}
        messages, truncated = truncate_tool_results(
            kwargs.get("messages", []), self._tool_result_limit
        )
        kwargs["messages"] = messages
        turn = self._recorder.start_turn(
            estimate_input_tokens(messages, kwargs.get("tools"))
        )
        turn.truncated_tool_results = truncated
        stream = await self._inner.create(**kwargs)
        return _InstrumentedStream(stream, self._recorder, turn)


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class InstrumentedClient:
    """Wraps an `AsyncOpenAI`-compatible client to record every model call.

    Sits outside the LangSmith-wrapped client, so tracing is unaffected. Each
    `chat.completions.create` call starts a new turn in `recorder`.

    Args:
        inner: The (LangSmith-wrapped) client to delegate to.
        recorder: Receives one `TurnRecord` per model call.
        extra: Arguments added to every call (e.g. `stream_options`).
        tool_result_limit: Maximum characters of each tool message sent to
            the model, or `None` for no cap.
    """

    def __init__(
        self,
        inner: Any,
        recorder: RunRecorder,
        extra: dict[str, Any] | None = None,
        tool_result_limit: int | None = None,
    ) -> None:
        self.chat = _Chat(
            _Completions(
                inner.chat.completions, recorder, extra or {}, tool_result_limit
            )
        )
