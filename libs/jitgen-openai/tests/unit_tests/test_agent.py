import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Self
from unittest.mock import MagicMock

import langsmith
import pytest
from langsmith import Client
from mcp.server.mcpserver import MCPServer
from mcp.shared.memory import create_client_server_memory_streams

from jitgen_openai import IptcAgent, McpToolBridge, PtcAgent
from jitgen_openai.agent import _close_truncated_json, _spread_object_argument

BOTH_AGENTS = pytest.mark.parametrize("agent_cls", [IptcAgent, PtcAgent])

# A plain fake with the streamed-chunk shape `IptcAgent` reads stands in for
# `AsyncOpenAI`; the MCP test serves a real server over in-memory streams.


@asynccontextmanager
async def _in_memory_transport(
    server: MCPServer,
) -> AsyncIterator[tuple[object, object]]:
    """Serve `server` over in-memory streams for the duration of the `with` block."""
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
            try:
                yield client_streams
            finally:
                task.cancel()


def _make_add_server() -> MCPServer:
    server = MCPServer("test")

    @server.tool()
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return server


@dataclass
class _FunctionDelta:
    arguments: str | None = None


@dataclass
class _ToolCallDelta:
    index: int = 0
    id: str | None = None
    function: _FunctionDelta | None = None


@dataclass
class _Delta:
    content: str | None = None
    tool_calls: list[_ToolCallDelta] | None = None


@dataclass
class _Choice:
    delta: _Delta


@dataclass
class _Chunk:
    choices: list[_Choice] = field(default_factory=list)

    @classmethod
    def content(cls, text: str) -> "_Chunk":
        return cls([_Choice(_Delta(content=text))])

    @classmethod
    def tool_call(cls, *, call_id: str | None, arguments: str | None) -> "_Chunk":
        return cls(
            [
                _Choice(
                    _Delta(
                        tool_calls=[
                            _ToolCallDelta(
                                id=call_id, function=_FunctionDelta(arguments)
                            )
                        ]
                    )
                )
            ]
        )


def _eval_call_chunks(call_id: str, code: str, *, chunk_size: int = 5) -> list[_Chunk]:
    """Split one `eval` tool call's `arguments` JSON into streamed chunks."""
    payload = json.dumps({"code": code})
    chunks = [
        _Chunk.tool_call(
            call_id=call_id if i == 0 else None,
            arguments=payload[i : i + chunk_size],
        )
        for i in range(0, len(payload), chunk_size)
    ]
    return chunks


class _FakeStream:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self) -> AsyncIterator[_Chunk]:
        return self._generate()

    async def _generate(self) -> AsyncIterator[_Chunk]:
        for chunk in self._chunks:
            # Give the executor's worker thread real time to run dispatched
            # statements between chunks, as network latency would.
            await asyncio.sleep(0.01)
            yield chunk

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


class _FakeCompletions:
    def __init__(self, turns: list[list[_Chunk]]) -> None:
        self._turns = turns
        self.calls: list[dict[str, object]] = []
        self.streams: list[_FakeStream] = []

    async def create(self, **kwargs: object) -> _FakeStream:
        self.calls.append(kwargs)
        stream = _FakeStream(self._turns[len(self.calls) - 1])
        self.streams.append(stream)
        return stream


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    """Stands in for `AsyncOpenAI`: same `.chat.completions.create(...)` shape."""

    def __init__(self, turns: list[list[_Chunk]]) -> None:
        self.chat = _FakeChat(_FakeCompletions(turns))


async def test_single_tool_call_then_final_answer():
    turns = [
        _eval_call_chunks("call_1", "print(1 + 1)"),
        [_Chunk.content("The "), _Chunk.content("answer is 2.")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[])

    answer = await agent.arun("what is 1 + 1?")

    assert answer == "The answer is 2."
    assert len(client.chat.completions.calls) == 2
    # First turn is forced to call `eval`; once it has, the model is free to answer.
    assert client.chat.completions.calls[0]["tool_choice"] == "required"
    assert client.chat.completions.calls[1]["tool_choice"] == "auto"


async def test_tool_result_is_fed_back_as_a_tool_message():
    turns = [
        _eval_call_chunks("call_1", "print(1 + 1)"),
        [_Chunk.content("done")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[])

    await agent.arun("what is 1 + 1?")

    second_call_messages = client.chat.completions.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "2\n"
    assert tool_messages[0]["tool_call_id"] == "call_1"


async def test_repl_state_persists_across_turns():
    """A variable set in one turn's code is still in scope in the next turn's."""
    turns = [
        _eval_call_chunks("call_1", "x = 10"),
        _eval_call_chunks("call_2", "print(x + 5)"),
        [_Chunk.content("15")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[])

    answer = await agent.arun("compute something in two steps")

    assert answer == "15"
    tool_messages = [
        m
        for call in client.chat.completions.calls
        for m in call["messages"]
        if m["role"] == "tool"
    ]
    assert tool_messages[-1]["content"] == "15\n"


async def test_execution_error_is_reported_back_instead_of_raised():
    turns = [
        _eval_call_chunks("call_1", "raise ValueError('boom')"),
        [_Chunk.content("it failed")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[])

    answer = await agent.arun("do something that fails")

    assert answer == "it failed"
    tool_messages = [
        m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"
    ]
    assert "boom" in tool_messages[0]["content"]


async def test_stream_is_aborted_and_closed_on_a_mid_stream_error():
    """A statement failing before the `code` argument finishes streaming
    should abort the model's own stream right there — early exit, so
    generation isn't paid for once the outcome is known — rather than
    reading the tool call out to its natural end regardless."""
    full_code = "raise ValueError('boom')\nprint('never reached')"
    turns = [
        _eval_call_chunks("call_1", full_code),
        [_Chunk.content("it failed")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[])

    await agent.arun("do something that fails partway through")

    assert client.chat.completions.streams[0].closed

    second_call_messages = client.chat.completions.calls[1]["messages"]
    assistant_messages = [m for m in second_call_messages if m["role"] == "assistant"]
    raw_arguments = assistant_messages[0]["tool_calls"][0]["function"]["arguments"]
    parsed = json.loads(raw_arguments)  # raises if unclosed/invalid
    assert full_code.startswith(parsed["code"])
    assert "raise ValueError('boom')" in parsed["code"]
    assert parsed["code"] != full_code  # aborted before the whole call streamed

    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert "boom" in tool_messages[0]["content"]


async def test_raises_after_max_turns_without_a_final_answer():
    turns = [_eval_call_chunks(f"call_{i}", "print(1)") for i in range(3)]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[], max_turns=3)

    with pytest.raises(RuntimeError, match="no final answer after 3 turns"):
        await agent.arun("loop forever")


async def test_executor_timeout_is_forwarded_to_the_executor():
    """A short `executor_timeout` interrupts a statement the default 60s wouldn't."""
    turns = [
        _eval_call_chunks("call_1", "import time\ntime.sleep(0.2)\nprint('done')"),
        [_Chunk.content("ok")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[], executor_timeout=0.01)

    await agent.arun("sleep past a very short timeout")

    tool_messages = [
        m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"
    ]
    assert "timed out after 0.01s" in tool_messages[0]["content"]


async def test_on_output_and_on_delta_callbacks_fire():
    turns = [
        _eval_call_chunks("call_1", "print('hi')"),
        [_Chunk.content("hi")],
    ]
    client = _FakeClient(turns)
    outputs: list[str] = []
    deltas: list[str] = []
    agent = IptcAgent(
        client,
        "fake-model",
        bridges=[],
        on_output=outputs.append,
        on_delta=deltas.append,
    )

    await agent.arun("say hi")

    assert "hi\n" in outputs
    assert "".join(deltas).startswith('{"code')


async def test_agent_calls_a_real_mcp_tool_through_generated_code():
    """The generated `eval` code calls a real MCP tool via a real bridge."""
    server = _make_add_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        turns = [
            _eval_call_chunks("call_1", "result = await add(a=2, b=3)\nprint(result)"),
            [_Chunk.content("The sum is 5.")],
        ]
        client = _FakeClient(turns)
        agent = IptcAgent(
            client,
            "fake-model",
            bridges=[bridge],
            system_prompt="You are a helpful agent.",
        )

        answer = await agent.arun("add 2 and 3 using the add tool")

    assert answer == "The sum is 5."
    first_call = client.chat.completions.calls[0]
    assert first_call["messages"][0] == {
        "role": "system",
        "content": "You are a helpful agent.",
    }
    code_description = first_call["tools"][0]["function"]["parameters"]["properties"][
        "code"
    ]["description"]
    assert "await add(a, b)" in code_description


async def test_chunks_without_choices_are_skipped():
    turns = [
        [_Chunk(), *_eval_call_chunks("call_1", "print(1)"), _Chunk()],
        [_Chunk(), _Chunk.content("done")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[])

    assert await agent.arun("print 1") == "done"


async def test_only_the_first_tool_call_is_executed():
    other_call = _Chunk.tool_call(call_id="call_2", arguments='{"code": "print(2)"}')
    other_call.choices[0].delta.tool_calls[0].index = 1
    turns = [
        [other_call, *_eval_call_chunks("call_1", "print(1)")],
        [_Chunk.content("done")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[])

    await agent.arun("print 1")

    messages = client.chat.completions.calls[1]["messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "call_1", "content": "1\n"}
    ]


@pytest.mark.parametrize(
    ("raw", "expected_code"),
    [
        ('{"code": "print(1)"}', "print(1)"),
        ('{"code": "print(1)', "print(1)"),
        ('{"code": "a\\', "a"),
        ('{"code": "a\\u00', "a"),
        ('{"code": "a\\"b', 'a"b'),
    ],
)
def test_close_truncated_json_yields_valid_json(raw: str, expected_code: str):
    assert json.loads(_close_truncated_json(raw)) == {"code": expected_code}


def test_close_truncated_json_leaves_unrecoverable_input_unchanged():
    assert _close_truncated_json('{"co') == '{"co'


async def test_agent_traces_eval_and_the_tool_calls_made_from_it(traced_runs):
    server = _make_add_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        turns = [
            _eval_call_chunks("call_1", "result = await add(a=2, b=3)\nprint(result)"),
            [_Chunk.content("5")],
        ]
        agent = IptcAgent(_FakeClient(turns), "fake-model", bridges=[bridge])
        await agent.arun("add 2 and 3")

    [agent_run] = traced_runs.named("IptcAgent")
    [eval_run] = traced_runs.named("eval")
    [tool_run] = traced_runs.named("add")
    assert agent_run["run_type"] == "chain"
    assert agent_run["inputs"] == {"task": "add 2 and 3"}
    assert agent_run["outputs"] == {"answer": "5"}
    assert eval_run["parent_run_id"] == agent_run["id"]
    assert eval_run["inputs"] == {"code": "result = await add(a=2, b=3)\nprint(result)"}
    assert eval_run["outputs"] == {"output": "5\n"}
    assert tool_run["parent_run_id"] == eval_run["id"]
    assert tool_run["inputs"] == {"a": 2, "b": 3}
    assert tool_run["outputs"] == {"output": "5"}


async def test_agent_traces_one_eval_run_per_turn(traced_runs):
    turns = [
        _eval_call_chunks("call_1", "x = 10"),
        _eval_call_chunks("call_2", "print(x + 5)"),
        [_Chunk.content("15")],
    ]
    agent = IptcAgent(_FakeClient(turns), "fake-model", bridges=[])

    await agent.arun("compute something in two steps")

    eval_runs = traced_runs.named("eval")
    assert [run["inputs"]["code"] for run in eval_runs] == ["x = 10", "print(x + 5)"]
    assert [run["outputs"]["output"] for run in eval_runs] == ["(no output)", "15\n"]


async def test_agent_traces_a_failed_eval_with_its_error(traced_runs):
    turns = [
        _eval_call_chunks("call_1", "raise ValueError('boom')\nprint('never')"),
        [_Chunk.content("it failed")],
    ]
    agent = IptcAgent(_FakeClient(turns), "fake-model", bridges=[])

    await agent.arun("fail")

    [eval_run] = traced_runs.named("eval")
    assert "boom" in eval_run["error"]
    assert eval_run["inputs"]["code"].startswith("raise ValueError('boom')")


async def test_agent_sends_no_runs_when_tracing_is_disabled():
    client = MagicMock(spec=Client)
    turns = [_eval_call_chunks("call_1", "print(1)"), [_Chunk.content("1")]]
    agent = IptcAgent(_FakeClient(turns), "fake-model", bridges=[])

    with langsmith.tracing_context(enabled=False, client=client):
        assert await agent.arun("print 1") == "1"

    assert client.method_calls == []


@BOTH_AGENTS
async def test_both_agents_run_the_same_tool_loop(agent_cls):
    turns = [
        _eval_call_chunks("call_1", "x = 10"),
        _eval_call_chunks("call_2", "print(x + 5)"),
        [_Chunk.content("15")],
    ]
    client = _FakeClient(turns)
    agent = agent_cls(client, "fake-model", bridges=[])

    assert await agent.arun("compute something in two steps") == "15"
    tool_messages = [
        m
        for call in client.chat.completions.calls
        for m in call["messages"]
        if m["role"] == "tool"
    ]
    assert tool_messages[-1]["content"] == "15\n"
    assert client.chat.completions.calls[0]["tool_choice"] == "required"


@BOTH_AGENTS
async def test_both_agents_produce_the_same_trace_tree(agent_cls, traced_runs):
    server = _make_add_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        turns = [
            _eval_call_chunks("call_1", "result = await add(a=2, b=3)\nprint(result)"),
            [_Chunk.content("5")],
        ]
        agent = agent_cls(_FakeClient(turns), "fake-model", bridges=[bridge])
        await agent.arun("add 2 and 3")

    [agent_run] = traced_runs.named(agent_cls.__name__)
    [eval_run] = traced_runs.named("eval")
    [tool_run] = traced_runs.named("add")
    assert agent_run["run_type"] == "chain"
    assert agent_run["outputs"] == {"answer": "5"}
    assert eval_run["parent_run_id"] == agent_run["id"]
    assert eval_run["outputs"] == {"output": "5\n"}
    assert eval_run["extra"]["metadata"]["ls_code_input_language"] == "python"
    assert tool_run["parent_run_id"] == eval_run["id"]


@pytest.mark.parametrize(
    ("agent_cls", "strategy"), [(IptcAgent, "incremental"), (PtcAgent, "buffered")]
)
async def test_agent_run_is_tagged_with_its_strategy(agent_cls, strategy, traced_runs):
    turns = [_eval_call_chunks("call_1", "print(1)"), [_Chunk.content("1")]]
    await agent_cls(_FakeClient(turns), "fake-model", bridges=[]).arun("print 1")

    [agent_run] = traced_runs.named(agent_cls.__name__)
    assert agent_run["extra"]["metadata"]["ptc_strategy"] == strategy


async def test_both_agents_report_the_same_error_for_the_same_code():
    results = []
    for agent_cls in (IptcAgent, PtcAgent):
        turns = [
            _eval_call_chunks("call_1", "raise ValueError('boom')"),
            [_Chunk.content("it failed")],
        ]
        client = _FakeClient(turns)
        await agent_cls(client, "fake-model", bridges=[]).arun("fail")
        messages = client.chat.completions.calls[1]["messages"]
        results.append(next(m["content"] for m in messages if m["role"] == "tool"))

    assert results[0] == results[1]
    assert "boom" in results[0]


_CHUNKED_CALL = _eval_call_chunks("call_1", "print(1)\nprint(2)\nx = 3\ny = 4\nz = 5")


async def _deltas_before_first_output(agent_cls: type) -> int:
    events: list[str] = []
    turns = [_CHUNKED_CALL, [_Chunk.content("done")]]
    agent = agent_cls(
        _FakeClient(turns),
        "fake-model",
        bridges=[],
        on_delta=lambda text: events.append("delta"),
        on_output=lambda output: events.append("output"),
    )
    await agent.arun("print twice")
    return events[: events.index("output")].count("delta")


async def test_ptc_agent_executes_only_after_the_tool_call_has_streamed():
    assert await _deltas_before_first_output(PtcAgent) == len(_CHUNKED_CALL)


async def test_iptc_agent_executes_while_the_tool_call_is_streaming():
    assert await _deltas_before_first_output(IptcAgent) < len(_CHUNKED_CALL)


async def test_ptc_agent_reads_the_whole_tool_call_despite_an_early_error(
    traced_runs,
):
    full_code = "raise ValueError('boom')\nprint('never reached')"
    turns = [_eval_call_chunks("call_1", full_code), [_Chunk.content("it failed")]]
    client = _FakeClient(turns)

    await PtcAgent(client, "fake-model", bridges=[]).arun("fail")

    messages = client.chat.completions.calls[1]["messages"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "code": full_code
    }
    [eval_run] = traced_runs.named("eval")
    assert eval_run["inputs"] == {"code": full_code}
    assert "boom" in eval_run["error"]


async def test_ptc_agent_reports_arguments_without_code():
    payload = json.dumps({"script": "print(1)"})
    turns = [
        [_Chunk.tool_call(call_id="call_1", arguments=payload)],
        [_Chunk.content("no code")],
    ]
    client = _FakeClient(turns)

    await PtcAgent(client, "fake-model", bridges=[]).arun("print 1")

    messages = client.chat.completions.calls[1]["messages"]
    tool_message = next(m for m in messages if m["role"] == "tool")
    assert "invalid `eval` arguments" in tool_message["content"]


@BOTH_AGENTS
async def test_completion_options_are_sent_with_every_model_call(agent_cls):
    turns = [_eval_call_chunks("call_1", "print(1)"), [_Chunk.content("1")]]
    client = _FakeClient(turns)
    agent = agent_cls(
        client,
        "fake-model",
        bridges=[],
        completion_options={"seed": 42, "temperature": 0.0},
    )

    await agent.arun("print 1")

    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert all(call["seed"] == 42 and call["temperature"] == 0.0 for call in calls)


@BOTH_AGENTS
def test_completion_options_cannot_override_agent_arguments(agent_cls):
    with pytest.raises(ValueError, match="cannot set: model, tool_choice"):
        agent_cls(
            _FakeClient([]),
            "fake-model",
            bridges=[],
            completion_options={"tool_choice": "none", "model": "other", "seed": 1},
        )


@BOTH_AGENTS
async def test_on_tool_result_receives_each_turns_result(agent_cls):
    turns = [
        _eval_call_chunks("call_1", "print(1)"),
        _eval_call_chunks("call_2", "raise ValueError('boom')"),
        [_Chunk.content("done")],
    ]
    results: list[str] = []
    agent = agent_cls(
        _FakeClient(turns), "fake-model", bridges=[], on_tool_result=results.append
    )

    await agent.arun("two turns")

    assert results[0] == "1\n"
    assert results[1].startswith("execution error:") and "boom" in results[1]
    assert len(results) == 2


BOTH_AGENTS_JS = pytest.mark.parametrize(
    "agent_cls", [IptcAgent, PtcAgent], ids=["iptc-js", "ptc-js"]
)


@BOTH_AGENTS_JS
async def test_javascript_agents_keep_state_across_turns(agent_cls):
    turns = [
        _eval_call_chunks("call_1", "const x = 10;"),
        _eval_call_chunks("call_2", "console.log(x + 5);"),
        [_Chunk.content("15")],
    ]
    client = _FakeClient(turns)
    agent = agent_cls(client, "fake-model", bridges=[], language="javascript")

    assert await agent.arun("compute something in two steps") == "15"
    last_messages = client.chat.completions.calls[-1]["messages"]
    tool_messages = [m for m in last_messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["(no output)", "15\n"]


@BOTH_AGENTS_JS
async def test_javascript_code_calls_an_mcp_tool_with_an_object_argument(
    agent_cls, traced_runs
):
    server = _make_add_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        code = "const result = await add({a: 2, b: 3});\nconsole.log(result);"
        turns = [_eval_call_chunks("call_1", code), [_Chunk.content("5")]]
        client = _FakeClient(turns)
        agent = agent_cls(client, "fake-model", bridges=[bridge], language="javascript")

        assert await agent.arun("add 2 and 3") == "5"

    tool_messages = [
        m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"
    ]
    assert tool_messages[0]["content"] == "5\n"
    [eval_run] = traced_runs.named("eval")
    [tool_run] = traced_runs.named("add")
    assert eval_run["extra"]["metadata"]["ls_code_input_language"] == "javascript"
    assert tool_run["parent_run_id"] == eval_run["id"]
    # The object argument is spread before tracing, so the trace shows kwargs.
    assert tool_run["inputs"] == {"a": 2, "b": 3}


async def test_javascript_eval_tool_describes_javascript_and_object_arguments():
    server = _make_add_server()
    async with McpToolBridge(lambda: _in_memory_transport(server)) as bridge:
        turns = [_eval_call_chunks("call_1", "console.log(1);"), [_Chunk.content("1")]]
        client = _FakeClient(turns)
        agent = IptcAgent(client, "fake-model", bridges=[bridge], language="javascript")
        await agent.arun("print 1")

    function = client.chat.completions.calls[0]["tools"][0]["function"]
    code_description = function["parameters"]["properties"]["code"]["description"]
    assert function["description"].startswith("Execute JavaScript code")
    assert "console.log" in function["description"]
    assert code_description.startswith("JavaScript source code to execute.")
    assert "await add({a, b})" in code_description
    assert "asyncio" not in code_description


async def test_javascript_error_is_reported_back_instead_of_raised():
    turns = [
        _eval_call_chunks("call_1", 'throw new Error("boom");'),
        [_Chunk.content("it failed")],
    ]
    client = _FakeClient(turns)
    agent = IptcAgent(client, "fake-model", bridges=[], language="javascript")

    assert await agent.arun("fail") == "it failed"
    tool_messages = [
        m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"
    ]
    assert tool_messages[0]["content"].startswith("execution error:")
    assert "boom" in tool_messages[0]["content"]


async def test_iptc_agent_executes_javascript_while_the_tool_call_is_streaming():
    code = "console.log(1);\nconsole.log(2);\nconst x = 3;\nconst y = 4;\nconst z = 5;"
    call = _eval_call_chunks("call_1", code)
    events: list[str] = []
    agent = IptcAgent(
        _FakeClient([call, [_Chunk.content("done")]]),
        "fake-model",
        bridges=[],
        language="javascript",
        on_delta=lambda text: events.append("delta"),
        on_output=lambda output: events.append("output"),
    )

    await agent.arun("print twice")

    assert events[: events.index("output")].count("delta") < len(call)


@BOTH_AGENTS
def test_unsupported_language_raises(agent_cls):
    with pytest.raises(ValueError, match="unsupported language: 'ruby'"):
        agent_cls(_FakeClient([]), "fake-model", bridges=[], language="ruby")


async def test_spread_object_argument_spreads_a_lone_dict_into_keywords():
    async def echo(*args: object, **kwargs: object) -> str:
        return f"{args} {kwargs}"

    spread = _spread_object_argument(echo)

    assert await spread({"a": 1, "b": 2}) == "() {'a': 1, 'b': 2}"
    assert await spread(1, 2) == "(1, 2) {}"
    assert await spread({"a": 1}, 2) == "({'a': 1}, 2) {}"
    assert await spread() == "() {}"
    assert spread.__name__ == "echo"


async def test_spread_object_argument_drops_omitted_types():
    async def echo(**kwargs: object) -> str:
        return str(kwargs)

    spread = _spread_object_argument(echo, omit=type(None))

    assert await spread({"a": 1, "b": None}) == "{'a': 1}"


@BOTH_AGENTS_JS
async def test_javascript_undefined_properties_are_not_sent_to_tools(agent_cls):
    received: list[dict[str, object]] = []

    async def record(**kwargs: object) -> str:
        received.append(kwargs)
        return "ok"

    class _Bridge:
        def __init__(self) -> None:
            self.callables = {"record": record}

        def describe_tools(self, language: str) -> str:
            return ""

    code = "const missing = {};\nawait record({a: 1, b: missing.b});"
    turns = [_eval_call_chunks("call_1", code), [_Chunk.content("done")]]
    agent = agent_cls(
        _FakeClient(turns), "fake-model", bridges=[_Bridge()], language="javascript"
    )

    await agent.arun("call a tool")

    assert received == [{"a": 1}]
