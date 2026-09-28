import json

import pytest
from conftest import PARCS_TOOLS, FakeParcs
from mcp.types import Tool

from iptc_parcs.bridge import EVAL_CODE_GUIDANCE, ExperimentBridge, openai_tool_schema
from iptc_parcs.config import Scenario
from iptc_parcs.faults import FaultInjector
from iptc_parcs.metrics import RunRecorder


def make_bridge(
    scenario: Scenario = Scenario.NATURAL,
) -> tuple[ExperimentBridge, RunRecorder]:
    recorder = RunRecorder()
    return ExperimentBridge(FakeParcs(), recorder, FaultInjector(scenario)), recorder


def test_run_layer_schema_keeps_only_truly_required_parameters():
    run_layer = next(t for t in PARCS_TOOLS if t.name == "run_layer")
    schema = openai_tool_schema(run_layer)["function"]["parameters"]

    assert schema["required"] == ["sessionId", "parallelism"]
    assert set(schema["properties"]) >= {"previousLayerId", "parameters", "datasetUrl"}


def test_schema_conversion_does_not_mutate_the_tool():
    tool = Tool(
        name="t",
        description="d",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    openai_tool_schema(tool)
    assert tool.input_schema["required"] == []


@pytest.mark.parametrize("language", ["python", "javascript"])
def test_describe_tools_appends_the_guidance_for_the_language(language):
    bridge, _ = make_bridge()
    description = bridge.describe_tools(language)

    assert description.startswith(f"- signatures in {language}")
    assert description.endswith(EVAL_CODE_GUIDANCE[language])


def test_guidance_is_written_for_each_language():
    python, javascript = EVAL_CODE_GUIDANCE["python"], EVAL_CODE_GUIDANCE["javascript"]

    assert "```python" in python and "json.loads" in python
    assert "```javascript" in javascript and "JSON.parse" in javascript
    assert "String.raw" in javascript
    assert "await create_session({sourceCode: source})" in javascript


async def test_calls_are_recorded_with_cluster_time():
    bridge, recorder = make_bridge()
    recorder.start_turn(est_input=0)

    reply = await bridge.callables["run_layer"](sessionId="s1", parallelism=2)

    assert json.loads(reply)["layerId"] == "l1"
    [call] = recorder.tool_calls
    assert (call.name, call.ok, call.turn) == ("run_layer", True, 0)
    assert call.total_elapsed_seconds == 4.5
    assert call.failure_count == 0


async def test_fault_compile_replaces_only_the_first_create_session():
    bridge, recorder = make_bridge(Scenario.FAULT_COMPILE)
    inner_calls = bridge._inner.calls

    first = await bridge.callables["create_session"](sourceCode="x")
    second = await bridge.callables["create_session"](sourceCode="x")

    assert "error" in json.loads(first)
    assert json.loads(second) == {"sessionId": "s1"}
    assert [c[0] for c in inner_calls] == ["create_session"]
    assert [(c.ok, c.injected) for c in recorder.tool_calls] == [
        (False, True),
        (True, False),
    ]


async def test_a_raising_tool_is_recorded_and_reraised():
    bridge, recorder = make_bridge()

    with pytest.raises(ConnectionError):
        await bridge.callables["get_cluster_info"]()

    [call] = recorder.tool_calls
    assert not call.ok
    assert "ConnectionError" in call.error
