# Adapted from misc/iptc-parcs/tests/unit_tests/conftest.py.
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import langsmith
import pytest

from iptc_bfcl.dataset import CATEGORIES, BfclEntry, load_entries


@dataclass
class FakeFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class FakeToolCall:
    index: int = 0
    id: str | None = None
    function: FakeFunction | None = None


@dataclass
class FakeDelta:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None
    reasoning_content: str | None = None


@dataclass
class FakeChoice:
    delta: FakeDelta
    finish_reason: str | None = None


@dataclass
class FakeUsageDetails:
    reasoning_tokens: int | None = None


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    completion_tokens_details: FakeUsageDetails | None = None


@dataclass
class FakeChunk:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage | None = None
    id: str = "gen-1"
    provider: str | None = None


def content_chunk(text: str, finish: str | None = None) -> FakeChunk:
    return FakeChunk([FakeChoice(FakeDelta(content=text), finish)])


def reasoning_chunk(text: str) -> FakeChunk:
    return FakeChunk([FakeChoice(FakeDelta(reasoning_content=text))])


def tool_chunk(
    arguments: str,
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    finish: str | None = None,
) -> FakeChunk:
    call = FakeToolCall(index, call_id, FakeFunction(name, arguments))
    return FakeChunk([FakeChoice(FakeDelta(tool_calls=[call]), finish)])


class FakeStream:
    """Like the LangSmith-traced stream: `__aiter__` is an async generator."""

    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[FakeChunk]:
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True

    async def close(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(self, turns: list[list[FakeChunk]]) -> None:
        self._turns = turns
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []

    async def create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        stream = FakeStream(self._turns[len(self.calls) - 1])
        self.streams.append(stream)
        return stream


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeClient:
    def __init__(self, turns: list[list[FakeChunk]]) -> None:
        self.chat = FakeChat(FakeCompletions(turns))


def eval_turn(code: str, *, chunk_size: int = 7) -> list[FakeChunk]:
    """One streamed `eval` tool call whose `code` argument is `code`."""
    payload = json.dumps({"code": code})
    return [
        tool_chunk(
            payload[i : i + chunk_size],
            call_id="call_1" if i == 0 else None,
            name="eval" if i == 0 else None,
        )
        for i in range(0, len(payload), chunk_size)
    ]


FIXTURES = Path(__file__).parent / "fixtures"


def load_entry(entry_id: str) -> BfclEntry:
    """One entry of the committed fixture subset, which mirrors BFCL's layout."""
    [entry] = [
        entry
        for category in CATEGORIES
        for entry in load_entries(category, FIXTURES)
        if entry.id == entry_id
    ]
    return entry


@pytest.fixture(autouse=True)
def _no_tracing() -> Iterator[None]:
    """Keep LangSmith off whatever the environment says."""
    with langsmith.tracing_context(enabled=False):
        yield
