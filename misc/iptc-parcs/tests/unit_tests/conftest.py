import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import langsmith
import pytest
from mcp.types import Tool


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


# The live PARCS server's tool list (names, descriptions, input schemas).
PARCS_TOOLS = [
    Tool(
        name=tool["name"],
        description=tool["description"],
        input_schema=tool["inputSchema"],
    )
    for tool in json.loads(
        (Path(__file__).parent / "fixtures" / "parcs_tools.json").read_text(
            encoding="utf-8"
        )
    )
]


class FakeParcs:
    """Duck-typed, unstarted `McpToolBridge` serving the live tool contract.

    `create_session` and `run_layer` answer like live PARCS and record their
    keyword arguments; `get_cluster_info` fails as a dropped connection does.
    """

    def __init__(self) -> None:
        self.tools = {tool.name: tool for tool in PARCS_TOOLS}
        self.calls: list[tuple[str, dict[str, Any]]] = []

        async def create_session(**kwargs: Any) -> str:
            self.calls.append(("create_session", kwargs))
            return json.dumps({"sessionId": "s1"})

        async def run_layer(**kwargs: Any) -> str:
            self.calls.append(("run_layer", kwargs))
            return json.dumps(
                {"layerId": "l1", "totalElapsedSeconds": 4.5, "failureCount": 0}
            )

        async def broken(**kwargs: Any) -> str:
            raise ConnectionError("MCP connection lost")

        self.callables = {
            "create_session": create_session,
            "run_layer": run_layer,
            "get_cluster_info": broken,
        }

    def describe_tools(self, language: str = "python") -> str:
        return f"- signatures in {language}"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_tracing() -> Iterator[None]:
    """Keep LangSmith off whatever the environment says."""
    with langsmith.tracing_context(enabled=False):
        yield
