from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from jitgen_openai import CodeLanguage, McpToolBridge
from jitgen_openai.mcp_bridge import _is_optional
from mcp.types import Tool

from iptc_parcs.faults import FaultInjector, parse_reply
from iptc_parcs.metrics import RunRecorder, ToolCallRecord

_PROMPTS = Path(__file__).parent / "prompts"
# Appended to the `eval` tool's description: how to write the code, per language.
EVAL_CODE_GUIDANCE: dict[str, str] = {
    "python": (_PROMPTS / "eval_code.md").read_text(encoding="utf-8"),
    "javascript": (_PROMPTS / "eval_code_js.md").read_text(encoding="utf-8"),
}


def openai_tool_schema(tool: Tool) -> dict[str, Any]:
    """Convert an MCP tool into an OpenAI function tool.

    Parameters the server lists as required but documents a default for are
    made optional, with the same rule `McpToolBridge.describe_tools` uses, so
    the baseline sees the same signatures as the code-writing strategies.

    Args:
        tool: The MCP tool.

    Returns:
        The OpenAI function-tool definition.
    """
    schema: dict[str, Any] = dict(tool.input_schema or {"type": "object"})
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    schema["required"] = [
        name
        for name, spec in properties.items()
        if not _is_optional(name, spec, required)
    ]
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


class ExperimentBridge:
    """The single tool surface every arm uses.

    Duck-types `McpToolBridge` (`callables`, `tools`, `describe_tools`), so it
    can be passed straight to `IptcAgent`/`PtcAgent`, and also serves the
    baseline agent. Every call goes through fault injection and is recorded.

    Args:
        inner: A started `McpToolBridge` to live PARCS.
        recorder: Receives one `ToolCallRecord` per call.
        faults: Decides which calls get an injected fault.
    """

    def __init__(
        self, inner: McpToolBridge, recorder: RunRecorder, faults: FaultInjector
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._faults = faults
        self.tools: dict[str, Tool] = inner.tools
        self.callables: dict[str, Callable[..., Awaitable[str]]] = {
            name: self._wrap(name, tool) for name, tool in inner.callables.items()
        }

    def describe_tools(self, language: CodeLanguage = "python") -> str:
        """Tool signatures plus the code guidance, for the `eval` description.

        Args:
            language: The language of the `eval` code.

        Returns:
            The signatures in `language`'s calling convention, followed by that
            language's guidance.
        """
        signatures = self._inner.describe_tools(language)
        return f"{signatures}\n\n{EVAL_CODE_GUIDANCE[language]}"

    def openai_tools(self) -> list[dict[str, Any]]:
        """The MCP tools as native OpenAI function tools, for the baseline."""
        return [openai_tool_schema(tool) for tool in self.tools.values()]

    def _wrap(
        self, name: str, tool: Callable[..., Awaitable[str]]
    ) -> Callable[..., Awaitable[str]]:
        recorder, faults = self._recorder, self._faults

        async def call(*args: Any, **kwargs: Any) -> str:
            start = recorder.now()
            turn = recorder.current_turn_index()
            injected = faults.fault_for(name)
            try:
                reply = (
                    injected if injected is not None else await tool(*args, **kwargs)
                )
            except BaseException as exc:
                recorder.tool_calls.append(
                    ToolCallRecord(
                        name,
                        start,
                        recorder.now(),
                        ok=False,
                        error=repr(exc),
                        turn=turn,
                    )
                )
                raise
            recorder.tool_calls.append(
                _record(name, start, recorder.now(), reply, injected, turn)
            )
            return reply

        call.__name__ = name
        return call


def _record(
    name: str,
    start: float,
    end: float,
    reply: str,
    injected: str | None,
    turn: int | None,
) -> ToolCallRecord:
    data = parse_reply(reply) or {}
    error = data.get("error") if isinstance(data.get("error"), str) else None
    if error is None and reply.startswith("[tool error]"):
        error = reply
    elapsed = data.get("totalElapsedSeconds")
    failures = data.get("failureCount")
    return ToolCallRecord(
        name,
        start,
        end,
        ok=error is None,
        error=error,
        injected=injected is not None,
        total_elapsed_seconds=float(elapsed)
        if isinstance(elapsed, int | float)
        else None,
        failure_count=int(failures) if isinstance(failures, int) else None,
        turn=turn,
    )
