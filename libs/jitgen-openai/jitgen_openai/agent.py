import json
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import langsmith
from jitgen import (
    ExecutionError,
    ExecutorBase,
    ExtractionError,
    InProcPythonExecutor,
    JitGenError,
    Session,
    StreamDriver,
    create_python_session,
)
from langsmith.utils import tracing_is_enabled
from openai import AsyncOpenAI

from ._tracing import EvalTracer
from .mcp_bridge import CodeLanguage, McpToolBridge
from .segmenter import OpenAIToolCallSegmenter

_EVAL_TOOL_NAME = "eval"
_CODE_PROPERTY = "code"
_MAX_ESCAPE_LENGTH = len("\\uXXXX")
_RESERVED_COMPLETION_OPTIONS = frozenset(
    {"model", "messages", "stream", "tools", "tool_choice"}
)


@dataclass
class _ToolCall:
    id: str
    arguments: str
    result: str


def _close_truncated_json(raw: str) -> str:
    """Close `arguments` that were cut off inside the `code` string.

    Some servers (e.g. LM Studio) re-parse tool-call arguments from the
    history and fail on truncated JSON. A cut inside the string can only
    leave a partial escape behind, so trimming a few characters and closing
    the object always finds a valid prefix.

    Args:
        raw: The `arguments` JSON received so far.

    Returns:
        `raw` if it is valid JSON, otherwise its longest prefix that becomes
        valid once closed with `"}`; `raw` unchanged if none does.
    """
    trims = range(min(_MAX_ESCAPE_LENGTH, len(raw)) + 1)
    candidates = [raw, *(raw[: len(raw) - trim] + '"}' for trim in trims)]
    for candidate in candidates:
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    return raw


def _describe(exc: BaseException) -> str:
    """Render `exc` as `"TypeError: ..."`, the way `Session` reports it.

    Args:
        exc: The exception to describe.

    Returns:
        `exc`'s type name, followed by `": {message}"` when it has one.
    """
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _decode_code(arguments: str) -> str:
    """Extract the `code` string from `eval` arguments, for tracing.

    Args:
        arguments: The (closed) `arguments` JSON.

    Returns:
        The decoded code, or `arguments` itself when it can't be decoded.
    """
    try:
        code = json.loads(arguments).get(_CODE_PROPERTY)
    except (json.JSONDecodeError, AttributeError):
        return arguments
    return code if isinstance(code, str) else arguments


def _spread_object_argument(
    tool: Callable[..., Awaitable[Any]], omit: type | tuple[type, ...] = ()
) -> Callable[..., Awaitable[Any]]:
    """Let JavaScript pass a tool's named arguments as one object.

    JavaScript has no keyword arguments, so `await fetch({url: "..."})`
    reaches Python as a single `dict` positional argument, which the bridge
    would otherwise bind whole to the tool's first parameter.

    Args:
        tool: The async tool callable exposed to the executed code.
        omit: Types of values to drop from the object, e.g. the engine's
            `undefined`: like `JSON.stringify`, a property read from a missing
            field (`{sessionId: session.sessionId}` after a failed call) is
            left out rather than sent as a value no tool can serialize.

    Returns:
        An async callable that spreads a lone `dict` argument into keywords
        and passes any other arguments through unchanged.
    """

    async def call(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs and isinstance(args[0], dict):
            named = {k: v for k, v in args[0].items() if not isinstance(v, omit)}
            return await tool(**named)
        return await tool(*args, **kwargs)

    call.__name__ = getattr(tool, "__name__", "tool")
    return call


def _python_executor(
    tools: Mapping[str, Callable[..., Awaitable[Any]]], timeout: float
) -> ExecutorBase:
    return InProcPythonExecutor(tools=tools, timeout=timeout)


def _javascript_executor(
    tools: Mapping[str, Callable[..., Awaitable[Any]]], timeout: float
) -> ExecutorBase:
    # Imported lazily so the Python path never loads QuickJS or ANTLR.
    from jitgen_js import QuickJsExecutor
    from quickjs_rs import Undefined

    spread_tools = {
        name: _spread_object_argument(tool, omit=Undefined)
        for name, tool in tools.items()
    }
    return QuickJsExecutor(tools=spread_tools, timeout=timeout)


def _python_session(executor: ExecutorBase) -> Session:
    return create_python_session(executor=executor)


def _javascript_session(executor: ExecutorBase) -> Session:
    from jitgen_js import create_javascript_session

    return create_javascript_session(executor=executor)


@dataclass(frozen=True)
class _Language:
    """Everything about running `eval` code that depends on its language."""

    name: CodeLanguage
    title: str
    output_hint: str
    helpers_hint: str
    make_executor: Callable[
        [Mapping[str, Callable[..., Awaitable[Any]]], float], ExecutorBase
    ]
    make_session: Callable[[ExecutorBase], Session]


_LANGUAGES: dict[str, _Language] = {
    "python": _Language(
        name="python",
        title="Python",
        output_hint="what it prints",
        helpers_hint=(
            "call each with `await`, directly at top level (no "
            "`async def main(): ...`/`asyncio.run(...)` wrapper)"
        ),
        make_executor=_python_executor,
        make_session=_python_session,
    ),
    "javascript": _Language(
        name="javascript",
        title="JavaScript",
        output_hint="what it logs with `console.log`",
        helpers_hint=(
            "call each with `await`, directly at top level (no "
            "`async function main() {...}` wrapper), passing its arguments as "
            "one object, e.g. `await name({arg: value})`"
        ),
        make_executor=_javascript_executor,
        make_session=_javascript_session,
    ),
}


def _build_eval_tool(language: _Language, tool_descriptions: str) -> dict[str, object]:
    """Build the `eval` tool schema, listing the helpers the code may call.

    Args:
        language: The language the `code` argument is written in.
        tool_descriptions: One line per available helper; empty when none.

    Returns:
        An OpenAI function-tool definition.
    """
    helpers_note = (
        "\n\nThe following async helpers are already available in this "
        f"environment; {language.helpers_hint} — do not define or import them "
        f"yourself:\n{tool_descriptions}"
        if tool_descriptions
        else ""
    )
    return {
        "type": "function",
        "function": {
            "name": _EVAL_TOOL_NAME,
            "description": (
                f"Execute {language.title} code and return {language.output_hint}. "
                "Use this for any computation or tool call instead of answering "
                "from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    _CODE_PROPERTY: {
                        "type": "string",
                        "description": (
                            f"{language.title} source code to execute.{helpers_note}"
                        ),
                    }
                },
                "required": [_CODE_PROPERTY],
            },
        },
    }


def _collect(
    new_outputs: list[str], outputs: list[str], on_output: Callable[[str], None]
) -> None:
    for output in new_outputs:
        outputs.append(output)
        on_output(output)


class _CodeRunner(Protocol):
    """Executes one turn's `eval` code; the only part the agents differ in."""

    async def apush(self, text: str, outputs: list[str]) -> JitGenError | None:
        """Handle one streamed `arguments` fragment.

        Args:
            text: The next `arguments` fragment.
            outputs: This turn's output, appended to in place.

        Returns:
            The first execution error so far, or `None`.
        """
        ...

    async def afinish(self, arguments: str, outputs: list[str]) -> JitGenError | None:
        """Finish executing once the tool call has fully streamed.

        Args:
            arguments: The complete `arguments` JSON.
            outputs: This turn's output, appended to in place.

        Returns:
            The execution error, or `None` on success.
        """
        ...

    async def areset(self) -> None:
        """Prepare for the next turn, keeping interpreter state."""
        ...


class _IncrementalRunner:
    """Executes each statement as soon as it is complete, while streaming."""

    def __init__(
        self, driver: StreamDriver, tracer: EvalTracer, on_output: Callable[[str], None]
    ) -> None:
        self._driver = driver
        self._tracer = tracer
        self._on_output = on_output

    async def apush(self, text: str, outputs: list[str]) -> JitGenError | None:
        self._tracer.start_eval()
        try:
            _collect(await self._driver.apush(text), outputs, self._on_output)
        except JitGenError as exc:
            return exc
        return self._driver.error

    async def afinish(self, arguments: str, outputs: list[str]) -> JitGenError | None:
        try:
            _collect(await self._driver.afinish(), outputs, self._on_output)
        except JitGenError as exc:
            return exc
        return None

    async def areset(self) -> None:
        await self._driver.areset()


class _BufferedRunner:
    """Executes the whole `code` in one call once the tool call has streamed."""

    def __init__(
        self,
        executor: ExecutorBase,
        tracer: EvalTracer,
        on_output: Callable[[str], None],
    ) -> None:
        self._executor = executor
        self._tracer = tracer
        self._on_output = on_output

    async def apush(self, text: str, outputs: list[str]) -> JitGenError | None:
        return None

    async def afinish(self, arguments: str, outputs: list[str]) -> JitGenError | None:
        self._tracer.start_eval()
        try:
            code = json.loads(arguments)[_CODE_PROPERTY]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return ExtractionError(f"invalid `eval` arguments: {exc}", source=arguments)
        try:
            result = await self._executor.aexecute(code)
        except Exception as exc:  # noqa: BLE001  # reported back, as `Session` does
            error = ExecutionError(_describe(exc), statement=code)
            error.__cause__ = exc
            return error
        if result.output:
            _collect([result.output], outputs, self._on_output)
        return None if result.success else ExecutionError.from_result(result)

    async def areset(self) -> None:
        return None


class _EvalAgent(ABC):
    """Multi-turn `eval` tool loop over MCP tools, shared by both agents.

    Each turn's `eval` output is sent back as the tool result until the model
    answers in plain text. One interpreter spans a whole `arun()` call, so
    variables persist across turns. Callers own the bridges' lifecycle.

    Args:
        client: Any `AsyncOpenAI`-compatible client.
        model: The model identifier to request.
        bridges: Started bridges whose tools the generated code may call.
        system_prompt: Optional system message for every `arun()` call.
        language: The language the model writes `eval` code in. JavaScript
            runs in QuickJS and needs the `jitgen-js` package; its code passes
            each tool's arguments as one object.
        max_turns: Model round trips allowed before giving up.
        executor_timeout: Seconds allowed per executed statement (per whole
            `eval` call for `PtcAgent`); raise it for slow tools. In
            JavaScript it bounds only the engine's own running time, not time
            spent awaiting tools, so it never cuts off a slow tool call.
        force_first_tool_call: Require a tool call on the first turn, so the
            model computes instead of answering from memory.
        on_delta: Called with each streamed text or `code` fragment.
        on_output: Called with each piece of `eval` output as it is produced.
        on_tool_result: Called with each turn's tool result as sent back to the
            model: the output, or `execution error: ...`.
        completion_options: Extra arguments for every model call, e.g.
            `{"seed": 42}` for reproducible comparisons.

    Raises:
        ValueError: If `language` is unsupported, or `completion_options` sets
            an argument the agent sets itself.
    """

    _STRATEGY: ClassVar[str]

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        bridges: Sequence[McpToolBridge],
        *,
        system_prompt: str | None = None,
        language: CodeLanguage = "python",
        max_turns: int = 8,
        executor_timeout: float = 60.0,
        force_first_tool_call: bool = True,
        on_delta: Callable[[str], None] | None = None,
        on_output: Callable[[str], None] | None = None,
        on_tool_result: Callable[[str], None] | None = None,
        completion_options: Mapping[str, Any] | None = None,
    ) -> None:
        if language not in _LANGUAGES:
            msg = f"unsupported language: {language!r}"
            raise ValueError(msg)
        reserved = _RESERVED_COMPLETION_OPTIONS.intersection(completion_options or {})
        if reserved:
            msg = f"completion_options cannot set: {', '.join(sorted(reserved))}"
            raise ValueError(msg)
        self._language = _LANGUAGES[language]
        self._client = client
        self._model = model
        self._bridges = list(bridges)
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._executor_timeout = executor_timeout
        self._force_first_tool_call = force_first_tool_call
        self._on_delta = on_delta or (lambda _: None)
        self._on_output = on_output or (lambda _: None)
        self._on_tool_result = on_tool_result or (lambda _: None)
        self._completion_options = dict(completion_options or {})

    async def arun(self, task: str) -> str:
        """Solve `task`, calling `eval` (and, through it, MCP tools) as needed.

        When LangSmith tracing is enabled, the run is traced as a chain named
        after the agent class and tagged with its `ptc_strategy`. Each turn's
        `eval` execution is a child tool run, and each tool called from the
        executed code is a child of that `eval` run. Model calls nest under the
        chain when `client` is wrapped with `langsmith.wrappers.wrap_openai`.

        Args:
            task: The user's task.

        Returns:
            The model's final plain-text answer.

        Raises:
            RuntimeError: If the model still calls `eval` after `max_turns`.
        """
        async with langsmith.trace(
            type(self).__name__,
            run_type="chain",
            inputs={"task": task},
            metadata={"ptc_strategy": self._STRATEGY},
        ) as run:
            tracer = EvalTracer(
                run if tracing_is_enabled() else None, self._language.name
            )
            answer = await self._run(task, tracer)
            run.end(outputs={"answer": answer})
            return answer

    @abstractmethod
    def _open_runner(
        self, executor: ExecutorBase, tracer: EvalTracer
    ) -> AbstractAsyncContextManager[_CodeRunner]:
        """Open the runner that executes each turn's code on `executor`."""

    async def _run(self, task: str, tracer: EvalTracer) -> str:
        """Run the turn loop for `arun`.

        Args:
            task: The user's task.
            tracer: Traces each turn's `eval` execution.

        Returns:
            The model's final plain-text answer.

        Raises:
            RuntimeError: If the model still calls `eval` after `max_turns`.
        """
        tools = {
            name: tracer.wrap_tool(name, tool)
            for bridge in self._bridges
            for name, tool in bridge.callables.items()
        }
        descriptions = [
            bridge.describe_tools(self._language.name) for bridge in self._bridges
        ]
        eval_tool = _build_eval_tool(
            self._language, "\n".join(filter(None, descriptions))
        )

        messages: list[dict[str, object]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": task})

        executor = self._language.make_executor(tools, self._executor_timeout)
        async with executor, self._open_runner(executor, tracer) as runner:
            for turn in range(self._max_turns):
                force_tool = turn == 0 and self._force_first_tool_call
                outcome = await self._stream_turn(
                    runner, eval_tool, messages, force_tool, tracer
                )
                if isinstance(outcome, str):
                    return outcome
                self._on_tool_result(outcome.result)
                messages += [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": outcome.id,
                                "type": "function",
                                "function": {
                                    "name": _EVAL_TOOL_NAME,
                                    "arguments": outcome.arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": outcome.id,
                        "content": outcome.result,
                    },
                ]
                await runner.areset()

        msg = f"{type(self).__name__}: no final answer after {self._max_turns} turns"
        raise RuntimeError(msg)

    async def _stream_turn(
        self,
        runner: _CodeRunner,
        eval_tool: dict[str, object],
        messages: list[dict[str, object]],
        force_tool: bool,
        tracer: EvalTracer,
    ) -> str | _ToolCall:
        """Stream one turn and execute its `eval` call through `runner`.

        The stream is abandoned as soon as a statement fails (P8), so no
        further generation is paid for.

        Args:
            runner: Executes the turn's code.
            eval_tool: The `eval` tool definition.
            messages: The conversation so far.
            force_tool: Whether this turn must call `eval`.
            tracer: Traces this turn's `eval` execution.

        Returns:
            The plain-text answer when the model made no tool call, otherwise
            the tool call with its execution result.
        """
        stream = await self._client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=messages,
            tools=[eval_tool],
            tool_choice="required" if force_tool else "auto",
            **self._completion_options,
        )
        tool_call_id = ""
        content: list[str] = []
        arguments: list[str] = []
        outputs: list[str] = []
        error: JitGenError | None = None

        async with stream:
            async for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta.content:
                    content.append(delta.content)
                    self._on_delta(delta.content)
                for call in delta.tool_calls or ():
                    if call.index != 0:
                        continue
                    tool_call_id = call.id or tool_call_id
                    text = call.function.arguments if call.function else None
                    if text:
                        arguments.append(text)
                        self._on_delta(text)
                        error = await runner.apush(text, outputs)
                if error:
                    break

        if not tool_call_id:
            return "".join(content)
        closed_arguments = _close_truncated_json("".join(arguments))
        if error is None:
            error = await runner.afinish(closed_arguments, outputs)
        result = (
            f"execution error: {error}"
            if error
            else ("".join(outputs) or "(no output)")
        )
        tracer.end_eval(
            _decode_code(closed_arguments), result, str(error) if error else None
        )
        return _ToolCall(tool_call_id, closed_arguments, result)


class IptcAgent(_EvalAgent):
    """Incremental-PTC agent: executes the `eval` code while it streams.

    Each statement of the streamed `code` argument runs as soon as it is
    complete, so tool latency overlaps generation, and the stream is abandoned
    at the first failing statement.
    """

    _STRATEGY = "incremental"

    @asynccontextmanager
    async def _open_runner(
        self, executor: ExecutorBase, tracer: EvalTracer
    ) -> AsyncGenerator[_CodeRunner]:
        async with self._language.make_session(executor) as session:
            driver = StreamDriver(
                session, OpenAIToolCallSegmenter(property_name=_CODE_PROPERTY)
            )
            yield _IncrementalRunner(driver, tracer, self._on_output)


class PtcAgent(_EvalAgent):
    """Regular-PTC agent, for demos and benchmarks against `IptcAgent` only.

    The conventional baseline: waits for the whole `eval` tool call, then runs
    its complete `code` in one executor call. Everything else (turn loop,
    executor, tools, tracing) is shared with `IptcAgent`, so their LangSmith
    traces differ only in when execution happens.
    """

    _STRATEGY = "buffered"

    @asynccontextmanager
    async def _open_runner(
        self, executor: ExecutorBase, tracer: EvalTracer
    ) -> AsyncGenerator[_CodeRunner]:
        yield _BufferedRunner(executor, tracer, self._on_output)
