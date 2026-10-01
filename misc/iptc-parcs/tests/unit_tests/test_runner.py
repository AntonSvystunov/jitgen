from typing import Self

import httpx
import openai
import pytest
from conftest import FakeClient, FakeParcs, content_chunk, eval_turn, tool_chunk

from iptc_parcs.bridge import EVAL_CODE_GUIDANCE
from iptc_parcs.config import (
    LANGUAGES,
    Arm,
    RunSettings,
    RunSpec,
    Scenario,
    Strategy,
    parse_model,
    plan_arms,
)
from iptc_parcs.instrumentation import InstrumentedClient
from iptc_parcs.metrics import RunRecorder, ToolCallRecord
from iptc_parcs.runner import Outcome, _classify_exception, build_row, grade, run_agent


def _status_error(code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    return openai.APIStatusError(
        "boom", response=httpx.Response(code, request=request), body=None
    )


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (TimeoutError(), "time_limit"),
        (RuntimeError("IptcAgent: no final answer after 12 turns"), "iteration_limit"),
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
                'predict stream returned an error: {"code":500,"message":"failed to decode"}',
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
        (ConnectionError("MCP connection lost"), "infra_error"),
        (ValueError("bug"), "agent_error"),
    ],
)
def test_only_real_outages_are_infra_errors(exc, status):
    assert _classify_exception(exc) == status


_PYTHON_CODE = """import json
session = json.loads(await create_session(sourceCode="x"))
layer = json.loads(await run_layer(sessionId=session["sessionId"], parallelism=2))
print(layer["layerId"])"""

_JAVASCRIPT_CODE = """const session = JSON.parse(await create_session({sourceCode: "x"}));
const layer = JSON.parse(await run_layer({sessionId: session.sessionId, parallelism: 2}));
console.log(layer.layerId);"""

_BASELINE_TURN = [
    tool_chunk('{"sourceCode": "x"}', call_id="c1", name="create_session"),
    tool_chunk(
        '{"sessionId": "s1", "parallelism": 2}', index=1, call_id="c2", name="run_layer"
    ),
]

_CORRECT_ANSWER = 'Done. {"var_99": 0.017601, "cvar_99": 0.020165}'


def _spec(arm: Arm, scenario: Scenario = Scenario.NATURAL) -> RunSpec:
    return RunSpec(
        parse_model("lmstudio:fake-model"),
        arm,
        scenario,
        rep=0,
        seed=1,
        order=0,
        parcs_url="http://unused",
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


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda arm: arm.label)
async def test_each_arm_calls_parcs_and_is_graded(arm):
    fake = FakeClient([_first_turn(arm), [content_chunk(_CORRECT_ANSWER)]])
    recorder = RunRecorder()
    parcs = FakeParcs()
    spec = _spec(arm)

    outcome = await run_agent(spec, parcs, InstrumentedClient(fake, recorder), recorder)

    assert (outcome.status, outcome.error) == ("no_answer", "")
    assert outcome.answer == _CORRECT_ANSWER
    # JavaScript passes one object; it reaches PARCS as keyword arguments.
    assert parcs.calls == [
        ("create_session", {"sourceCode": "x"}),
        ("run_layer", {"sessionId": "s1", "parallelism": 2}),
    ]
    assert [(c.name, c.ok, c.turn) for c in recorder.tool_calls] == [
        ("create_session", True, 0),
        ("run_layer", True, 0),
    ]
    assert recorder.turns[0].made_tool_call
    assert not recorder.turns[0].execution_error
    row = build_row(spec, "2026-09-27T00:00:00+00:00", outcome, recorder)
    assert (row["status"], row["language"], row["iterations"]) == (
        "answered_correct",
        arm.language,
        2,
    )


@pytest.mark.parametrize("language", LANGUAGES)
async def test_code_arms_describe_the_tools_in_their_language(language):
    fake = FakeClient([_first_turn(Arm(Strategy.IPTC, language)), [content_chunk("")]])
    recorder = RunRecorder()

    await run_agent(
        _spec(Arm(Strategy.IPTC, language)),
        FakeParcs(),
        InstrumentedClient(fake, recorder),
        recorder,
    )

    [eval_tool] = fake.chat.completions.calls[0]["tools"]
    code = eval_tool["function"]["parameters"]["properties"]["code"]["description"]
    assert f"- signatures in {language}" in code
    assert code.endswith(EVAL_CODE_GUIDANCE[language])


@pytest.mark.parametrize("language", LANGUAGES)
async def test_a_failing_statement_is_an_execution_error(language):
    code = (
        "raise ValueError('boom')"
        if language == "python"
        else "throw new Error('boom');"
    )
    fake = FakeClient([eval_turn(code), [content_chunk("gave up")]])
    recorder = RunRecorder()

    outcome = await run_agent(
        _spec(Arm(Strategy.IPTC, language)),
        FakeParcs(),
        InstrumentedClient(fake, recorder),
        recorder,
    )

    assert outcome.answer == "gave up"
    assert recorder.turns[0].execution_error
    assert "boom" in recorder.turns[0].tool_result


async def test_fault_compile_is_injected_in_javascript_too():
    arm = Arm(Strategy.PTC, "javascript")
    fake = FakeClient([_first_turn(arm), [content_chunk(_CORRECT_ANSWER)]])
    recorder = RunRecorder()
    parcs = FakeParcs()

    await run_agent(
        _spec(arm, Scenario.FAULT_COMPILE),
        parcs,
        InstrumentedClient(fake, recorder),
        recorder,
    )

    # The injected compile error has no `sessionId`, so JS reads `undefined`,
    # which is left out of the call as `JSON.stringify` would.
    assert [(c.name, c.injected) for c in recorder.tool_calls] == [
        ("create_session", True),
        ("run_layer", False),
    ]
    assert parcs.calls[0] == ("run_layer", {"parallelism": 2})


async def test_an_unreachable_parcs_is_an_infra_error():
    class Unreachable(FakeParcs):
        async def __aenter__(self) -> Self:
            raise ConnectionError("refused")

    recorder = RunRecorder()
    outcome = await run_agent(
        _spec(Arm(Strategy.IPTC, "python")),
        Unreachable(),
        InstrumentedClient(FakeClient([]), recorder),
        recorder,
    )

    assert outcome.status == "infra_error"
    assert "refused" in outcome.error


@pytest.mark.parametrize(
    ("answer", "finish_reason", "status"),
    [
        (_CORRECT_ANSWER, None, "answered_correct"),
        ('{"var_99": 0.02, "cvar_99": 0.020165}', None, "answered_wrong"),
        ("", "length", "output_truncated"),
        ("no numbers", None, "no_answer"),
    ],
)
def test_grade(answer, finish_reason, status):
    recorder = RunRecorder()
    recorder.start_turn(est_input=0).finish_reason = finish_reason

    graded = grade(Outcome(answer, "no_answer"), recorder, tolerance=0.01)

    assert graded["status"] == status
    assert graded["correct"] is (status == "answered_correct")


def test_a_dropped_connection_turns_a_wrong_answer_into_an_infra_error():
    recorder = RunRecorder()
    recorder.tool_calls.append(
        ToolCallRecord("run_layer", 0.0, 1.0, ok=False, error="ConnectionError()")
    )

    graded = grade(Outcome("no numbers", "no_answer"), recorder, tolerance=0.01)

    assert (graded["status"], graded["infra_incidents"]) == ("infra_error", 1)


@pytest.mark.parametrize(
    "arm", [Arm(Strategy.BASELINE), Arm(Strategy.IPTC, "javascript")], ids=str
)
async def test_reasoning_effort_and_provider_reach_every_model_call(arm):
    spec = RunSpec(
        parse_model("openrouter:fake-model"),
        arm,
        Scenario.NATURAL,
        rep=0,
        seed=1,
        order=0,
        parcs_url="http://unused",
        settings=RunSettings(reasoning_effort="low", upstream=("deepinfra",)),
    )
    fake = FakeClient([_first_turn(arm), [content_chunk(_CORRECT_ANSWER)]])
    recorder = RunRecorder()

    await run_agent(spec, FakeParcs(), InstrumentedClient(fake, recorder), recorder)

    expected = {
        "reasoning": {"effort": "low"},
        "provider": {"order": ["deepinfra"], "allow_fallbacks": False},
    }
    assert [call["extra_body"] for call in fake.chat.completions.calls] == [
        expected,
        expected,
    ]
