import json
import time

import pytest
from conftest import FakeClient, content_chunk, load_entry, tool_chunk

from iptc_bfcl.baseline import ToolCallingAgent
from iptc_bfcl.metrics import RunRecorder
from iptc_bfcl.tools import BfclToolBridge


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


def _bridge(delay: float = 0.0) -> tuple[BfclToolBridge, RunRecorder]:
    recorder = RunRecorder()
    recorder.start_turn(est_input=0)
    return BfclToolBridge(load_entry("parallel_0"), recorder, delay), recorder


def _two_calls_turn() -> list:
    return [
        tool_chunk("", call_id="c1", name="spotify_play"),
        tool_chunk('{"artist": "Taylor Swift", '),
        tool_chunk('"duration": 20}'),
        tool_chunk(
            '{"artist": "Maroon 5", "duration": 15}',
            index=1,
            call_id="c2",
            name="spotify_play",
        ),
    ]


async def test_a_turns_calls_run_concurrently_and_return_in_call_order():
    bridge, recorder = _bridge(delay=0.2)
    client = FakeClient([_two_calls_turn(), [content_chunk("done")]])
    results = []

    started = time.monotonic()
    answer = await _agent(client, bridge, results).arun("task")
    elapsed = time.monotonic() - started

    assert answer == "done"
    assert elapsed < 0.35  # one delay, not two
    calls = client.chat.completions.calls
    assert calls[0]["tool_choice"] == "required"
    assert calls[1]["tool_choice"] == "auto"
    assert calls[0]["seed"] == 1
    assert [t["function"]["name"] for t in calls[0]["tools"]] == ["spotify_play"]
    messages = calls[1]["messages"]
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [m["tool_call_id"] for m in messages[3:]] == ["c1", "c2"]
    assert [json.loads(m["content"])["arguments"]["artist"] for m in messages[3:]] == [
        "Taylor Swift",
        "Maroon 5",
    ]
    assert [failed for _, failed in results] == [False, False]
    assert [c.name for c in recorder.tool_calls] == ["spotify.play"] * 2


async def test_bad_arguments_become_tool_errors():
    bridge, _ = _bridge()
    client = FakeClient(
        [
            [
                tool_chunk("{not json", call_id="c1", name="spotify_play"),
                tool_chunk(
                    '{"artist": "x"}', index=1, call_id="c2", name="spotify_play"
                ),
                tool_chunk("{}", index=2, call_id="c3", name="nope"),
            ],
            [content_chunk("gave up")],
        ]
    )
    results = []

    assert await _agent(client, bridge, results).arun("task") == "gave up"

    assert [failed for _, failed in results] == [True, True, True]
    errors = sorted(result for result, _ in results)
    assert any("invalid JSON" in e for e in errors)
    assert any("missing required argument(s): duration" in e for e in errors)
    assert any("unknown tool 'nope'" in e for e in errors)


async def test_raises_after_max_turns():
    bridge, _ = _bridge()
    turn = [
        tool_chunk('{"artist": "a", "duration": 1}', call_id="c", name="spotify_play")
    ]
    client = FakeClient([turn, turn, turn])

    with pytest.raises(RuntimeError, match="no final answer after 3 turns"):
        await _agent(client, bridge, []).arun("task")
