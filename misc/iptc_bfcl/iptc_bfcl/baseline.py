# Adapted from misc/iptc-parcs/iptc_parcs/baseline.py; a turn's calls run concurrently.
import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import langsmith
from langsmith.utils import tracing_is_enabled

from iptc_bfcl.tools import BfclToolBridge


@dataclass
class _PendingCall:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)


class ToolCallingAgent:
    """Baseline: the model calls the functions directly, without writing code.

    Mirrors `jitgen_openai`'s `_EvalAgent` loop (same messages, first turn
    forced to call a tool, same turn limit and completion options, same
    LangSmith shape); only the tool surface differs. The tool calls of one
    turn run concurrently, since a model emits them together only when they
    are independent, and each result goes back as its own `tool` message, in
    the order the calls were made.

    Args:
        client: Any `AsyncOpenAI`-compatible client.
        model: The model identifier to request.
        bridge: The experiment's tool surface.
        system_prompt: System message for every run.
        max_turns: Model round trips allowed before giving up.
        completion_options: Extra arguments for every model call.
        on_tool_result: Called with each tool result sent back to the model,
            and whether the call failed.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        bridge: BfclToolBridge,
        *,
        system_prompt: str,
        max_turns: int = 8,
        completion_options: Mapping[str, Any] | None = None,
        on_tool_result: Callable[[str, bool], None] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._bridge = bridge
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._completion_options = dict(completion_options or {})
        self._on_tool_result = on_tool_result or (lambda _result, _failed: None)

    async def arun(self, task: str) -> str:
        """Solve `task` by calling the tools until the model answers in text.

        Raises:
            RuntimeError: If the model still calls tools after `max_turns`.
        """
        async with langsmith.trace(
            type(self).__name__,
            run_type="chain",
            inputs={"task": task},
            metadata={"ptc_strategy": "baseline"},
        ) as run:
            answer = await self._run(task, trace=tracing_is_enabled())
            run.end(outputs={"answer": answer})
            return answer

    async def _run(self, task: str, *, trace: bool) -> str:
        tools = self._bridge.openai_tools()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task},
        ]
        for turn in range(self._max_turns):
            content, calls = await self._stream_turn(messages, tools, force=turn == 0)
            if not calls:
                return content
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": "".join(call.arguments),
                            },
                        }
                        for call in calls
                    ],
                }
            )
            results = await asyncio.gather(
                *(self._execute(call, trace=trace) for call in calls)
            )
            messages.extend(
                {"role": "tool", "tool_call_id": call.id, "content": result}
                for call, result in zip(calls, results)
            )
        msg = f"{type(self).__name__}: no final answer after {self._max_turns} turns"
        raise RuntimeError(msg)

    async def _stream_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        force: bool,
    ) -> tuple[str, list[_PendingCall]]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=messages,
            tools=tools,
            tool_choice="required" if force else "auto",
            **self._completion_options,
        )
        content: list[str] = []
        pending: dict[int, _PendingCall] = {}
        async with stream:
            async for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta.content:
                    content.append(delta.content)
                for call in delta.tool_calls or ():
                    entry = pending.setdefault(call.index, _PendingCall())
                    entry.id = call.id or entry.id
                    if call.function is not None:
                        entry.name = call.function.name or entry.name
                        if call.function.arguments:
                            entry.arguments.append(call.function.arguments)
        calls = [pending[i] for i in sorted(pending) if pending[i].name]
        return "".join(content), calls

    async def _execute(self, call: _PendingCall, *, trace: bool) -> str:
        raw = "".join(call.arguments) or "{}"
        tool = self._bridge.callables.get(call.name)
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            return self._report(
                f"tool error: invalid JSON arguments: {exc}", failed=True
            )
        if tool is None:
            return self._report(f"tool error: unknown tool {call.name!r}", failed=True)
        if not isinstance(arguments, dict):
            return self._report(
                "tool error: arguments must be a JSON object", failed=True
            )
        try:
            if trace:
                async with langsmith.trace(
                    call.name, run_type="tool", inputs=arguments
                ) as run:
                    result = await tool(**arguments)
                    run.end(outputs={"output": result})
            else:
                result = await tool(**arguments)
        except (TypeError, RuntimeError, ConnectionError) as exc:
            return self._report(f"tool error: {exc}", failed=True)
        return self._report(result, failed=False)

    def _report(self, result: str, *, failed: bool) -> str:
        self._on_tool_result(result, failed)
        return result
