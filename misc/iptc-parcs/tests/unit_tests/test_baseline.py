import json

import pytest
from conftest import FakeClient, content_chunk, tool_chunk
from test_bridge import make_bridge

from iptc_parcs.baseline import ToolCallingAgent


def _agent(client, bridge, results):
    return ToolCallingAgent(
        client,
        "fake-model",
        bridge,
        system_prompt="system",
        max_turns=3,
        completion_options={"seed": 1},
        on_tool_result=lambda result, failed: results.append((result, failed)),
    )


async def test_tool_calls_are_executed_and_fed_back():
    bridge, recorder = make_bridge()
    client = FakeClient(
        [
            [
                tool_chunk("", call_id="c1", name="create_session"),
                tool_chunk('{"sourceCode": '),
                tool_chunk('"x"}'),
                tool_chunk(
                    '{"sessionId": "s1", "parallelism": 2}',
                    index=1,
                    call_id="c2",
                    name="run_layer",
                ),
            ],
            [content_chunk('done {"var_99": 1, "cvar_99": 2}')],
        ]
    )
    results = []

    answer = await _agent(client, bridge, results).arun("task")

    assert answer == 'done {"var_99": 1, "cvar_99": 2}'
    calls = client.chat.completions.calls
    assert calls[0]["tool_choice"] == "required"
    assert calls[1]["tool_choice"] == "auto"
    assert calls[0]["seed"] == 1
    assert {t["function"]["name"] for t in calls[0]["tools"]} >= {"run_layer"}
    messages = calls[1]["messages"]
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert json.loads(messages[3]["content"]) == {"sessionId": "s1"}
    assert [name for name, _ in bridge._inner.calls] == ["create_session", "run_layer"]
    assert [failed for _, failed in results] == [False, False]
    assert [c.name for c in recorder.tool_calls] == ["create_session", "run_layer"]


async def test_bad_arguments_and_raising_tools_become_tool_errors():
    bridge, _ = make_bridge()
    client = FakeClient(
        [
            [
                tool_chunk("{not json", call_id="c1", name="create_session"),
                tool_chunk("{}", index=1, call_id="c2", name="get_cluster_info"),
            ],
            [content_chunk("gave up")],
        ]
    )
    results = []

    assert await _agent(client, bridge, results).arun("task") == "gave up"

    assert [failed for _, failed in results] == [True, True]
    assert "invalid JSON" in results[0][0]
    assert "MCP connection lost" in results[1][0]


async def test_raises_after_max_turns():
    bridge, _ = make_bridge()
    turn = [tool_chunk('{"sourceCode": "x"}', call_id="c", name="create_session")]
    client = FakeClient([turn, turn, turn])

    with pytest.raises(RuntimeError, match="no final answer after 3 turns"):
        await _agent(client, bridge, []).arun("task")
