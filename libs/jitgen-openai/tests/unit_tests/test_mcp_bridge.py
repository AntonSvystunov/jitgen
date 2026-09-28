import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import Tool

from jitgen_openai import McpToolBridge
from jitgen_openai.mcp_bridge import _describe_tool

# A real MCP server runs over in-memory streams: real protocol, no network.


def _new_threads(before: set[threading.Thread]) -> list[threading.Thread]:
    """Threads started since `before`, ignoring asyncio's default-executor workers."""
    return [
        thread
        for thread in set(threading.enumerate()) - before
        if not thread.name.startswith("asyncio")
    ]


@asynccontextmanager
async def _in_memory_transport(
    server: MCPServer, server_tasks: list[asyncio.Task[None]] | None = None
) -> AsyncIterator[tuple[object, object]]:
    """Serve `server` over in-memory streams for the duration of the `with` block.

    Each connection's server task is appended to `server_tasks`, so a test can
    cancel it to simulate the connection dropping.
    """
    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        server_read, server_write = server_streams

        async def run_server() -> None:
            await server._lowlevel_server.run(
                server_read,
                server_write,
                server._lowlevel_server.create_initialization_options(),
            )

        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(run_server())
            if server_tasks is not None:
                server_tasks.append(task)
            try:
                yield client_streams
            finally:
                task.cancel()


def _make_server() -> MCPServer:
    server = MCPServer("test")

    @server.tool()
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool()
    async def fail() -> str:
        """Always raises."""
        raise ValueError("kaboom")

    return server


async def test_start_discovers_tools_and_binds_callables():
    server = _make_server()
    bridge = McpToolBridge(lambda: _in_memory_transport(server))

    await bridge.start()
    try:
        assert set(bridge.tools) == {"add", "fail"}
        assert set(bridge.callables) == {"add", "fail"}
        assert await bridge.callables["add"](a=2, b=3) == "5"
    finally:
        await bridge.aclose()


async def test_describe_tools_renders_signature_and_description():
    server = _make_server()
    bridge = McpToolBridge(lambda: _in_memory_transport(server))

    await bridge.start()
    try:
        description = bridge.describe_tools()
        assert "- `await add(a, b)`: Add two numbers." in description
        assert "- `await fail()`: Always raises." in description
    finally:
        await bridge.aclose()


def test_describe_tool_marks_params_without_required_as_optional():
    tool = Tool(
        name="search",
        description="Search.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    )
    assert _describe_tool(tool) == "- `await search(query, limit=...)`: Search."


def test_describe_tool_treats_documented_defaults_as_optional():
    """The C# MCP SDK marks every parameter required but documents defaults."""
    tool = Tool(
        name="run_layer",
        description="Run.",
        input_schema={
            "type": "object",
            "properties": {
                "sessionId": {"type": "string", "description": "Session ID."},
                "previousLayerId": {
                    "type": "string",
                    "description": "Previous layer. (Default value: null)",
                },
            },
            "required": ["sessionId", "previousLayerId"],
        },
    )
    assert _describe_tool(tool) == (
        "- `await run_layer(sessionId, previousLayerId=...)`: Run."
    )


async def test_tool_execution_error_is_returned_not_raised():
    server = _make_server()
    bridge = McpToolBridge(lambda: _in_memory_transport(server))

    await bridge.start()
    try:
        result = await bridge.callables["fail"]()
        assert result.startswith("[tool error]")
    finally:
        await bridge.aclose()


async def test_callable_accepts_positional_arguments():
    """Generated code often calls a single-argument tool positionally."""
    server = _make_server()
    bridge = McpToolBridge(lambda: _in_memory_transport(server))

    await bridge.start()
    try:
        assert await bridge.callables["add"](2, 3) == "5"
    finally:
        await bridge.aclose()


async def test_positional_and_keyword_overlap_raises_type_error():
    server = _make_server()
    bridge = McpToolBridge(lambda: _in_memory_transport(server))

    await bridge.start()
    try:
        with pytest.raises(TypeError, match="multiple values"):
            await bridge.callables["add"](2, a=3, b=4)
    finally:
        await bridge.aclose()


async def test_usable_as_an_async_context_manager():
    server = _make_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        assert await bridge.callables["add"](a=10, b=20) == "30"


async def test_callable_works_from_another_threads_event_loop():
    """The executor awaits tools inside its own `asyncio.run()` on a worker thread."""
    server = _make_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        result = await asyncio.to_thread(asyncio.run, bridge.callables["add"](1, 2))
        assert result == "3"


async def test_concurrent_calls_all_complete():
    server = _make_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        add = bridge.callables["add"]
        results = await asyncio.gather(*(add(a=i, b=i) for i in range(5)))
    assert results == ["0", "2", "4", "6", "8"]


async def test_start_raises_when_connecting_fails_and_stops_its_thread():
    @asynccontextmanager
    async def broken_transport() -> AsyncIterator[tuple[object, object]]:
        raise ConnectionError("no server")
        yield  # pragma: no cover

    threads_before = set(threading.enumerate())
    bridge = McpToolBridge(broken_transport)

    with pytest.raises(ConnectionError, match="no server"):
        await asyncio.wait_for(bridge.start(), timeout=5)
    assert _new_threads(threads_before) == []


async def test_constructing_a_bridge_starts_no_thread():
    threads_before = set(threading.enumerate())
    McpToolBridge(lambda: _in_memory_transport(_make_server()))
    assert _new_threads(threads_before) == []


async def test_aclose_stops_the_bridge_thread_and_calls_then_fail():
    server = _make_server()
    threads_before = set(threading.enumerate())
    bridge = McpToolBridge(lambda: _in_memory_transport(server))
    await bridge.start()
    add = bridge.callables["add"]

    await bridge.aclose()

    assert _new_threads(threads_before) == []
    with pytest.raises(RuntimeError, match="not started"):
        await add(a=1, b=2)


async def test_aclose_without_start_is_a_no_op():
    await McpToolBridge(lambda: _in_memory_transport(_make_server())).aclose()


async def test_starting_twice_raises():
    server = _make_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        with pytest.raises(RuntimeError, match="already started"):
            await bridge.start()


async def test_dropped_connection_fails_one_call_then_reconnects():
    server = _make_server()
    server_tasks: list[asyncio.Task[None]] = []
    async with McpToolBridge(
        lambda: _in_memory_transport(server, server_tasks)
    ) as bridge:
        add = bridge.callables["add"]
        assert await add(a=1, b=1) == "2"

        server_tasks[-1].cancel()
        await asyncio.sleep(0.1)

        with pytest.raises(ConnectionError, match="not retried"):
            await add(a=2, b=2)
        assert await add(a=3, b=3) == "6"
    assert len(server_tasks) == 2


async def test_failed_reconnect_raises_connection_error():
    server = _make_server()
    server_tasks: list[asyncio.Task[None]] = []
    connections = 0

    def transport() -> AbstractAsyncContextManager[tuple[object, object]]:
        nonlocal connections
        connections += 1
        if connections > 1:
            return _broken_transport()
        return _in_memory_transport(server, server_tasks)

    async with McpToolBridge(transport) as bridge:
        server_tasks[-1].cancel()
        await asyncio.sleep(0.1)

        with pytest.raises(ConnectionError, match="reconnecting failed"):
            await bridge.callables["add"](a=1, b=1)


@asynccontextmanager
async def _broken_transport() -> AsyncIterator[tuple[object, object]]:
    raise ConnectionError("no server")
    yield  # pragma: no cover


def test_describe_tool_renders_javascript_object_arguments():
    tool = Tool(
        name="search",
        description="Search.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    )
    assert _describe_tool(tool, "javascript") == (
        "- `await search({query, limit?})`: Search."
    )


def test_describe_tool_renders_javascript_without_parameters():
    tool = Tool(name="ping", description="Ping.", input_schema={"type": "object"})
    assert _describe_tool(tool, "javascript") == "- `await ping()`: Ping."


async def test_describe_tools_accepts_a_language():
    server = _make_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        description = bridge.describe_tools("javascript")

    assert "- `await add({a, b})`: Add two numbers." in description
