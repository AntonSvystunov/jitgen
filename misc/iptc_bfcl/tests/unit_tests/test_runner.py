from dataclasses import replace

import httpx
import openai
import pytest
from conftest import FakeClient, content_chunk, eval_turn, load_entry, tool_chunk

from iptc_bfcl.config import (
    LANGUAGES,
    Arm,
    RunSettings,
    RunSpec,
    Strategy,
    parse_model,
    plan_arms,
)
from iptc_bfcl.instrumentation import InstrumentedClient
from iptc_bfcl.metrics import RunRecorder
from iptc_bfcl.runner import (
    SYSTEM_PROMPT,
    Outcome,
    _classify_exception,
    _SingleResponseClient,
    build_agent,
    build_row,
    grade,
    run_agent,
)
from iptc_bfcl.tools import EVAL_CODE_GUIDANCE, BfclToolBridge


def _status_error(code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    return openai.APIStatusError(
        "boom", response=httpx.Response(code, request=request), body=None
    )


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (TimeoutError(), "time_limit"),
        (RuntimeError("IptcAgent: no final answer after 4 turns"), "iteration_limit"),
        (
            openai.APIError(
                'returned 400: {"type":"exceed_context_size_error"}',
                request=None,
                body=None,
            ),
            "context_overflow",
        ),
        (
            openai.APIError(
                'predict stream returned an error: {"code":500,"message":"x"}',
                request=None,
                body=None,
            ),
            "infra_error",
        ),
        (
            openai.APIError("returned 400: bad request", request=None, body=None),
            "agent_error",
        ),
        (_status_error(503), "infra_error"),
        (_status_error(400), "agent_error"),
        (ValueError("bug"), "agent_error"),
    ],
)
def test_only_real_outages_are_infra_errors(exc, status):
    assert _classify_exception(exc) == status


# parallel_multiple_1: the area of a 7x3 rectangle and of a circle of radius 5.
# Both functions take floats; the code passes ints, as JavaScript always does.
_PYTHON_CODE = """rect = await area_rectangle_calculate(length=7, breadth=3)
print(rect)
circle = await area_circle_calculate(radius=5)
print(circle)"""

_JAVASCRIPT_CODE = """const rect = await area_rectangle_calculate({length: 7, breadth: 3});
console.log(rect);
const circle = await area_circle_calculate({radius: 5});
console.log(circle);"""

_BASELINE_TURN = [
    tool_chunk(
        '{"length": 7, "breadth": 3}', call_id="c1", name="area_rectangle_calculate"
    ),
    tool_chunk('{"radius": 5}', index=1, call_id="c2", name="area_circle_calculate"),
]


def _spec(arm: Arm, entry_id: str = "parallel_multiple_1", **settings) -> RunSpec:
    return RunSpec(
        parse_model("lmstudio:fake-model"),
        arm,
        load_entry(entry_id),
        rep=0,
        seed=1,
        order=0,
        settings=RunSettings(tool_delay=0.01, **settings),
    )


def _first_turn(arm: Arm) -> list:
    if arm.strategy is Strategy.BASELINE:
        return _BASELINE_TURN
    return eval_turn(_PYTHON_CODE if arm.language == "python" else _JAVASCRIPT_CODE)


ALL_ARMS = plan_arms(list(Strategy), list(LANGUAGES))


def test_every_strategy_runs_in_both_languages():
    assert [arm.label for arm in ALL_ARMS] == [
        "baseline",
        "ptc-python",
        "ptc-javascript",
        "iptc-python",
        "iptc-javascript",
    ]


@pytest.mark.parametrize("full_cycle", [False, True], ids=["single", "full"])
@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda arm: arm.label)
async def test_each_arm_calls_the_functions_and_is_graded(arm, full_cycle):
    fake = FakeClient([_first_turn(arm), [content_chunk("Done.")]])
    recorder = RunRecorder()
    spec = _spec(arm, full_cycle=full_cycle)

    outcome = await run_agent(spec, InstrumentedClient(fake, recorder), recorder)

    # A single-response run stops once the first response's calls finish.
    answer = "Done." if full_cycle else ""
    assert (outcome.status, outcome.error, outcome.answer) == ("finished", "", answer)
    assert len(fake.chat.completions.calls) == (2 if full_cycle else 1)
    calls = sorted((c.name, c.arguments, c.ok, c.turn) for c in recorder.tool_calls)
    assert calls == [
        ("area_circle.calculate", '{"radius": 5}', True, 0),
        ("area_rectangle.calculate", '{"breadth": 3, "length": 7}', True, 0),
    ]
    assert all(c.end - c.start >= 0.01 for c in recorder.tool_calls)
    assert recorder.turns[0].made_tool_call
    assert not recorder.turns[0].execution_error
    assert fake.chat.completions.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT
    row = build_row(spec, "2026-10-01T00:00:00+00:00", outcome, recorder)
    assert row["status"] == "answered_correct"
    assert (row["correct"], row["first_turn_correct"]) == (True, True)
    assert (row["expected_calls"], row["successful_calls"]) == (2, 2)
    assert (row["entry_id"], row["language"], row["iterations"]) == (
        "parallel_multiple_1",
        arm.language,
        2 if full_cycle else 1,
    )
    assert row["tool_delay"] == 0.01
    assert row["mode"] == ("full_cycle" if full_cycle else "single_response")
    assert row["wall_seconds"] >= recorder.tool_calls[-1].end


@pytest.mark.parametrize("language", LANGUAGES)
async def test_code_arms_describe_the_functions_in_their_language(language):
    arm = Arm(Strategy.IPTC, language)
    fake = FakeClient([_first_turn(arm), [content_chunk("")]])
    recorder = RunRecorder()

    await run_agent(_spec(arm), InstrumentedClient(fake, recorder), recorder)

    [eval_tool] = fake.chat.completions.calls[0]["tools"]
    code = eval_tool["function"]["parameters"]["properties"]["code"]["description"]
    assert "area_circle_calculate" in code
    assert "- `radius` (number, required)" in code
    assert code.endswith(EVAL_CODE_GUIDANCE[language])


@pytest.mark.parametrize("language", LANGUAGES)
async def test_iptc_stops_at_an_invalid_call(language):
    code = (
        "a = await area_circle_calculate(diameter=5)\nb = await area_circle_calculate(radius=5)"
        if language == "python"
        else "const a = await area_circle_calculate({diameter: 5});\n"
        "const b = await area_circle_calculate({radius: 5});"
    )
    fake = FakeClient([eval_turn(code), [content_chunk("gave up")]])
    recorder = RunRecorder()

    outcome = await run_agent(
        _spec(Arm(Strategy.IPTC, language)),
        InstrumentedClient(fake, recorder),
        recorder,
    )

    assert (outcome.status, outcome.answer) == ("finished", "")
    assert len(fake.chat.completions.calls) == 1
    assert recorder.turns[0].execution_error
    assert "unexpected argument(s): diameter" in recorder.turns[0].tool_result
    assert [c.ok for c in recorder.tool_calls] == [False]


async def test_the_entry_system_message_follows_the_harness_prompt():
    arm = Arm(Strategy.BASELINE)
    spec = _spec(arm, "live_simple_58-27-0")
    fake = FakeClient([[content_chunk("no call")]])
    recorder = RunRecorder()

    await run_agent(spec, InstrumentedClient(fake, recorder), recorder)

    sent = fake.chat.completions.calls[0]["messages"]
    assert sent[0]["content"] == f"{SYSTEM_PROMPT}\n\n{spec.entry.system}"
    assert sent[1] == {"role": "user", "content": spec.entry.user}


async def test_a_run_over_its_time_limit_is_a_time_limit():
    arm = Arm(Strategy.PTC, "python")
    spec = replace(_spec(arm), settings=RunSettings(tool_delay=5.0, time_limit=0.05))
    fake = FakeClient([_first_turn(arm), [content_chunk("Done.")]])
    recorder = RunRecorder()

    outcome = await run_agent(spec, InstrumentedClient(fake, recorder), recorder)

    assert outcome.status == "time_limit"


@pytest.mark.parametrize(
    ("outcome_status", "finish_reason", "with_calls", "status"),
    [
        ("finished", None, True, "answered_correct"),
        ("finished", None, False, "answered_wrong"),
        ("finished", "length", False, "output_truncated"),
        ("iteration_limit", None, True, "iteration_limit"),
    ],
)
async def test_grade(outcome_status, finish_reason, with_calls, status):
    entry = load_entry("parallel_multiple_1")
    recorder = RunRecorder()
    recorder.start_turn(est_input=0).finish_reason = finish_reason
    if with_calls:
        bridge = BfclToolBridge(entry, recorder, delay=0.0)
        await bridge.callables["area_rectangle_calculate"](length=7, breadth=3)
        await bridge.callables["area_circle_calculate"](radius=5)

    graded = grade(Outcome("text", outcome_status), recorder, entry)

    assert graded["status"] == status
    assert graded["correct"] is with_calls


@pytest.mark.parametrize(
    "arm", [Arm(Strategy.BASELINE), Arm(Strategy.IPTC, "javascript")], ids=str
)
async def test_reasoning_effort_and_provider_reach_every_model_call(arm):
    spec = replace(
        _spec(arm),
        model=parse_model("openrouter:fake-model"),
        settings=RunSettings(
            reasoning_effort="low", upstream=("deepinfra",), full_cycle=True
        ),
    )
    fake = FakeClient([_first_turn(arm), [content_chunk("Done.")]])
    recorder = RunRecorder()

    await run_agent(spec, InstrumentedClient(fake, recorder), recorder)

    expected = {
        "reasoning": {"effort": "low"},
        "provider": {"order": ["deepinfra"], "allow_fallbacks": False},
    }
    assert [call["extra_body"] for call in fake.chat.completions.calls] == [
        expected,
        expected,
    ]


async def test_a_full_cycle_run_still_ends_at_its_iteration_limit():
    arm = Arm(Strategy.BASELINE)
    turn = _first_turn(arm)
    fake = FakeClient([turn, turn])
    recorder = RunRecorder()
    spec = _spec(arm, full_cycle=True, max_iterations=2)

    outcome = await run_agent(spec, InstrumentedClient(fake, recorder), recorder)

    assert outcome.status == "iteration_limit"


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda arm: arm.label)
async def test_a_single_response_agent_returns_instead_of_raising(arm):
    # A raised turn limit would mark every LangSmith trace as failed.
    fake = FakeClient([_first_turn(arm)])
    recorder = RunRecorder()
    spec = _spec(arm)
    client = _SingleResponseClient(InstrumentedClient(fake, recorder))
    bridge = BfclToolBridge(spec.entry, recorder, spec.settings.tool_delay)

    answer = await build_agent(spec, client, bridge, recorder).arun(spec.entry.user)

    assert answer == ""
    assert len(fake.chat.completions.calls) == 1
    assert len(recorder.turns) == 1
