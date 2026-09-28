import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Self

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, CallToolResult, TextContent, Tool

Transport = Callable[[], AbstractAsyncContextManager[tuple[Any, Any]]]
CodeLanguage = Literal["python", "javascript"]

_THREAD_JOIN_TIMEOUT = 5.0


def _render_result(result: CallToolResult) -> str:
    """Flatten a tool result's text content into one string.

    Args:
        result: The result of an MCP `tools/call` request.

    Returns:
        The concatenated text, prefixed with `[tool error]` for a tool-level
        failure (which MCP reports as a result, not an exception).
    """
    texts = [part.text for part in result.content if isinstance(part, TextContent)]
    text = "\n".join(texts) if texts else f"(non-text tool result: {result.content!r})"
    return f"[tool error] {text}" if result.is_error else text


def _is_optional(name: str, spec: dict[str, Any], required: set[str]) -> bool:
    """Whether a tool parameter may be omitted.

    The C# MCP SDK lists every parameter as required and only mentions a
    default in the description, e.g. `(Default value: null)`.

    Args:
        name: The parameter name.
        spec: The parameter's JSON schema.
        required: The schema's required parameter names.

    Returns:
        `True` when the parameter is not required or documents a default.
    """
    return name not in required or "(Default value:" in spec.get("description", "")


def _describe_tool(tool: Tool, language: CodeLanguage = "python") -> str:
    """Render one tool as a `- await name(params): description` line.

    Args:
        tool: A tool advertised by the server.
        language: The language of the generated code. JavaScript has no
            keyword arguments, so there a tool takes one object of named
            arguments instead.

    Returns:
        The one-line summary. Optional parameters are shown as `name=...` in
        Python and as `name?` in JavaScript, e.g. `fetch({url, max_length?})`.
    """
    schema = tool.input_schema or {}
    required = set(schema.get("required", []))
    optional_suffix = "=..." if language == "python" else "?"
    params = ", ".join(
        f"{name}{optional_suffix}" if _is_optional(name, spec, required) else name
        for name, spec in schema.get("properties", {}).items()
    )
    if language == "javascript" and params:
        params = f"{{{params}}}"
    return f"- `await {tool.name}({params})`: {tool.description or ''}"


def _bind_arguments(
    name: str, param_names: list[str], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Map positional and keyword arguments onto a tool's parameter names.

    Args:
        name: The tool name, for error messages.
        param_names: The tool's parameters, in declaration order.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        The tool-call arguments by parameter name.

    Raises:
        TypeError: On too many positional arguments, or a parameter given
            both positionally and by keyword.
    """
    if len(args) > len(param_names):
        msg = (
            f"{name}() takes {len(param_names)} argument(s) but {len(args)} were given"
        )
        raise TypeError(msg)
    arguments = dict(zip(param_names, args))
    overlap = sorted(arguments.keys() & kwargs.keys())
    if overlap:
        msg = f"{name}() got multiple values for argument(s): {', '.join(overlap)}"
        raise TypeError(msg)
    return arguments | kwargs


class McpToolBridge:
    """Exposes every tool of one MCP server as an `async def name(...) -> str`.

    Pass `bridge.callables` to `InProcPythonExecutor(tools=...)` so generated
    code can `await tool(arg=value)` (or pass arguments positionally). Agents
    running JavaScript spread a single object argument into keywords, so there
    generated code calls `await tool({arg: value})`.

    The executor runs each statement in its own short-lived event loop, while
    an MCP session must stay on the loop and task that opened it. So the
    session lives on a dedicated thread's loop for the bridge's whole life,
    and each tool call is submitted to that loop. If the connection drops,
    the failing call raises `ConnectionError` and the bridge reconnects for
    the next call.

    Args:
        transport: Returns a fresh async context manager yielding
            `(read_stream, write_stream)`, like `mcp`'s `stdio_client`; see the
            `stdio`/`streamable_http`/`sse` constructors.
    """

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._closing: asyncio.Event | None = None
        self._serving: asyncio.Task[None] | None = None
        self._reconnect_lock: asyncio.Lock | None = None
        self.tools: dict[str, Tool] = {}
        self.callables: dict[str, Callable[..., Awaitable[str]]] = {}

    @classmethod
    def stdio(cls, params: StdioServerParameters) -> "McpToolBridge":
        """Build a bridge to a local server launched over stdio.

        Args:
            params: How to launch the server.

        Returns:
            An unstarted bridge.
        """
        return cls(lambda: stdio_client(params))

    @classmethod
    def streamable_http(cls, url: str) -> "McpToolBridge":
        """Build a bridge to a remote server over streamable HTTP.

        Args:
            url: The server's MCP endpoint.

        Returns:
            An unstarted bridge.
        """
        return cls(lambda: streamable_http_client(url))

    @classmethod
    def sse(cls, url: str) -> "McpToolBridge":
        """Build a bridge to a remote server over legacy HTTP+SSE.

        Args:
            url: The server's SSE endpoint (typically ending in `/sse`).

        Returns:
            An unstarted bridge.
        """
        return cls(lambda: sse_client(url))

    async def start(self) -> None:
        """Connect, list the server's tools, and bind one callable per tool.

        Raises:
            RuntimeError: If the bridge is already started.
        """
        if self._loop is not None:
            msg = "McpToolBridge is already started"
            raise RuntimeError(msg)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="McpToolBridge", daemon=True
        )
        self._thread.start()
        self._reconnect_lock = asyncio.Lock()
        try:
            tools = await self._on_loop(self._open_session())
        except BaseException:
            await self._stop_loop()
            raise
        self.tools = {tool.name: tool for tool in tools}
        self.callables = {name: self._make_callable(name) for name in self.tools}

    async def aclose(self) -> None:
        """Close the session and stop the bridge's thread; a no-op if not started."""
        if self._loop is None:
            return
        try:
            await self._on_loop(self._close_session())
        finally:
            await self._stop_loop()

    def describe_tools(self, language: CodeLanguage = "python") -> str:
        """Render every tool's signature and description, one per line.

        Args:
            language: The language whose calling convention the signatures
                follow.

        Returns:
            One `- await name(params): description` line per tool.
        """
        return "\n".join(_describe_tool(tool, language) for tool in self.tools.values())

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _on_loop[T](self, coroutine: Coroutine[Any, Any, T]) -> Awaitable[T]:
        """Run `coroutine` on the bridge's loop, awaitable from any loop."""
        assert self._loop is not None
        return asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        )

    async def _serve(self, ready: asyncio.Future[list[Tool]]) -> None:
        """Hold one MCP session open until `_close_session()`."""
        try:
            async with (
                self._transport() as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                tools = (await session.list_tools()).tools
                self._session = session
                self._closing = asyncio.Event()
                ready.set_result(tools)
                await self._closing.wait()
        except BaseException as exc:
            if not ready.done():
                if isinstance(exc, asyncio.CancelledError):
                    ready.cancel()
                else:
                    ready.set_exception(exc)
            raise
        finally:
            self._session = None

    async def _open_session(self) -> list[Tool]:
        """Start serving a new session; on the bridge's loop."""
        ready: asyncio.Future[list[Tool]] = asyncio.get_running_loop().create_future()
        self._serving = asyncio.create_task(self._serve(ready))
        return await ready

    async def _close_session(self) -> None:
        """Close the current session, if any; on the bridge's loop."""
        task, self._serving = self._serving, None
        if task is None:
            return
        if self._closing is not None:
            self._closing.set()
        with contextlib.suppress(Exception):  # closing a dead connection may fail
            await task

    async def _reconnect(self, dead: ClientSession | None) -> None:
        """Replace the session `dead`, unless another call already did."""
        assert self._reconnect_lock is not None
        async with self._reconnect_lock:
            if self._session is not None and self._session is not dead:
                return
            await self._close_session()
            await self._open_session()

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Call a tool; on the bridge's loop.

        Raises:
            ConnectionError: If the connection was lost. The bridge reconnects,
                but never retries the call, since tools need not be idempotent.
        """
        session = self._session
        cause: MCPError | None = None
        if session is not None:
            try:
                return await session.call_tool(name, arguments)
            except MCPError as exc:
                if exc.code != CONNECTION_CLOSED:
                    raise
                cause = exc
        try:
            await asyncio.shield(self._reconnect(session))
        except Exception as exc:
            msg = f"MCP connection lost and reconnecting failed: {exc}"
            raise ConnectionError(msg) from exc
        msg = (
            "MCP connection lost; reconnected, but this call was not retried. "
            f"Call {name}() again if needed."
        )
        raise ConnectionError(msg) from cause

    async def _stop_loop(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        await asyncio.to_thread(thread.join, _THREAD_JOIN_TIMEOUT)
        if not thread.is_alive():
            loop.close()
        self._loop = self._thread = self._serving = self._closing = None

    def _make_callable(self, name: str) -> Callable[..., Awaitable[str]]:
        param_names = list((self.tools[name].input_schema or {}).get("properties", {}))

        async def call(*args: Any, **kwargs: Any) -> str:
            arguments = _bind_arguments(name, param_names, args, kwargs)
            if self._loop is None:
                msg = "McpToolBridge is not started"
                raise RuntimeError(msg)
            return _render_result(await self._on_loop(self._call_tool(name, arguments)))

        call.__name__ = name
        return call
