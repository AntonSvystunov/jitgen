import asyncio
import json
from dataclasses import replace

import pytest
from conftest import load_entry

from iptc_bfcl.metrics import RunRecorder
from iptc_bfcl.tools import (
    EVAL_CODE_GUIDANCE,
    SIMULATED_NOTE,
    BfclToolBridge,
    echo_result,
    sanitize_name,
    to_json_schema,
)


def _bridge(entry_id: str, delay: float = 0.0) -> tuple[BfclToolBridge, RunRecorder]:
    recorder = RunRecorder()
    recorder.start_turn(est_input=0)
    return BfclToolBridge(load_entry(entry_id), recorder, delay), recorder


def test_bfcl_types_become_json_schema_recursively():
    schema = {
        "type": "dict",
        "properties": {
            "x": {"type": "float"},
            "pair": {"type": "tuple", "items": {"type": "integer"}},
            "anything": {"type": "any", "description": "d"},
            "nested": {
                "type": "array",
                "items": {"type": "dict", "properties": {"y": {"type": "float"}}},
            },
        },
        "required": ["x"],
    }

    assert to_json_schema(schema) == {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "pair": {"type": "array", "items": {"type": "integer"}},
            "anything": {"description": "d"},
            "nested": {
                "type": "array",
                "items": {"type": "object", "properties": {"y": {"type": "number"}}},
            },
        },
        "required": ["x"],
    }


def test_dotted_names_are_sanitized_and_mapped_back():
    bridge, _ = _bridge("parallel_multiple_1")

    assert sanitize_name("math.factorial") == "math_factorial"
    assert bridge.bfcl_names["area_rectangle_calculate"] == "area_rectangle.calculate"
    assert {t["function"]["name"] for t in bridge.openai_tools()} == set(
        bridge.callables
    )


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("a.b", "collides"),
        ("x" * 65, "longer than 64"),
        ("JSON", "shadows"),
        ("print", "shadows"),
    ],
)
def test_names_no_arm_could_use_are_rejected(name, problem):
    entry = load_entry("simple_0")
    function = {**entry.functions[0], "name": name}
    functions = (
        (function, {**function, "name": "a_b"}) if name == "a.b" else (function,)
    )

    with pytest.raises(ValueError, match=problem):
        BfclToolBridge(replace(entry, functions=functions), RunRecorder(), 0.0)


def test_builtins_the_code_never_needs_may_be_shadowed():
    entry = load_entry("simple_0")
    function = {**entry.functions[0], "name": "sum"}

    bridge = BfclToolBridge(replace(entry, functions=(function,)), RunRecorder(), 0.0)

    assert list(bridge.callables) == ["sum"]


@pytest.mark.parametrize(
    ("language", "signature"),
    [
        ("python", "`await calculate_triangle_area(base, height, unit=...)`"),
        ("javascript", "`await calculate_triangle_area({base, height, unit?})`"),
    ],
)
def test_code_arms_see_every_parameter_doc(language, signature):
    bridge, _ = _bridge("simple_0")

    text = bridge.describe_tools(language)

    assert signature in text
    assert "- `base` (integer, required): The base of the triangle." in text
    assert "- `unit` (string, optional): The unit of measure" in text
    assert text.endswith(EVAL_CODE_GUIDANCE[language])


def test_nested_fields_enums_and_defaults_are_described():
    entry = load_entry("simple_0")
    schema = {
        "type": "dict",
        "properties": {
            "conditions": {
                "type": "dict",
                "description": "Filters.",
                "properties": {
                    "mode": {"type": "string", "enum": ["a", "b"], "default": "a"}
                },
            }
        },
        "required": [],
    }
    function = {"name": "f", "description": "F.", "parameters": schema}
    bridge = BfclToolBridge(replace(entry, functions=(function,)), RunRecorder(), 0.0)

    text = bridge.describe_tools("python")

    assert "- `conditions` (object, optional): Filters." in text
    assert '  - `mode` (string, optional, one of "a", "b", default "a")' in text


async def test_a_call_binds_positionals_waits_records_and_echoes(monkeypatch):
    slept = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    bridge, recorder = _bridge("parallel_multiple_1", delay=0.25)

    result = await bridge.callables["area_rectangle_calculate"](7, breadth=3.0)

    assert slept == [0.25]
    [record] = recorder.tool_calls
    assert (record.name, record.ok, record.turn) == (
        "area_rectangle.calculate",
        True,
        0,
    )
    assert json.loads(record.arguments) == {"length": 7, "breadth": 3.0}
    assert result == echo_result(
        "area_rectangle.calculate", {"length": 7, "breadth": 3.0}
    )
    assert json.loads(result)["status"] == "ok"
    assert json.loads(result)["note"] == SIMULATED_NOTE


async def test_tuples_are_recorded_as_lists():
    bridge, recorder = _bridge("simple_89")

    await bridge.callables["db_fetch_records"](
        database_name="StudentDB", table_name=("students",), conditions={}
    )

    assert json.loads(recorder.tool_calls[0].arguments)["table_name"] == ["students"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base": 1}, "missing required argument"),
        ({"base": 1, "height": 2, "colour": "red"}, "unexpected argument"),
    ],
)
async def test_invalid_calls_fail_like_a_real_signature(kwargs, message):
    bridge, recorder = _bridge("simple_0", delay=10.0)

    with pytest.raises(TypeError, match=message):
        await bridge.callables["calculate_triangle_area"](**kwargs)

    [record] = recorder.tool_calls
    assert not record.ok
    assert message in record.error


async def test_a_cancelled_call_is_recorded_as_failed():
    bridge, recorder = _bridge("simple_0", delay=10.0)
    task = asyncio.ensure_future(bridge.callables["calculate_triangle_area"](1, 2))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    [record] = recorder.tool_calls
    assert not record.ok
    assert "CancelledError" in record.error


async def test_concurrent_calls_overlap_their_delays():
    bridge, recorder = _bridge("parallel_0", delay=0.05)
    play = bridge.callables["spotify_play"]

    await asyncio.gather(play(artist="a", duration=1), play(artist="b", duration=2))

    first, second = recorder.tool_calls
    assert second.start < first.end
