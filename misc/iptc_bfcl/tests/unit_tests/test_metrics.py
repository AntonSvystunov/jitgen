# Adapted from misc/iptc-parcs/tests/unit_tests/test_metrics.py.
import pytest

from iptc_bfcl.metrics import RunRecorder, ToolCallRecord, TurnRecord, summarize


def _recorder() -> RunRecorder:
    recorder = RunRecorder()
    recorder.turns = [
        # Turn 0: streams 0-10s; its tool call fails at 4-5s.
        TurnRecord(
            0,
            0.0,
            end=10.0,
            completed=False,
            est_input=100,
            est_output=40,
            made_tool_call=True,
            execution_error=True,
        ),
        # Turn 1: streams 10-20s; its two tool calls run 15-30s, succeed.
        TurnRecord(
            1,
            10.0,
            end=20.0,
            completed=True,
            est_input=150,
            est_output=60,
            made_tool_call=True,
        ),
        # Turn 2: final answer, 30-35s.
        TurnRecord(2, 30.0, end=35.0, completed=True, est_input=200, est_output=20),
    ]
    recorder.tool_calls = [
        ToolCallRecord("f", 4.0, 5.0, ok=False, error="TypeError: bad", turn=0),
        ToolCallRecord("f", 15.0, 16.0, ok=True, turn=1),
        ToolCallRecord("g", 16.0, 30.0, ok=True, turn=1),
    ]
    return recorder


def test_iterations_are_split_into_failed_self_correction_and_successful():
    metrics = summarize(_recorder(), wall_seconds=36.0)

    assert metrics["iterations"] == 3
    assert metrics["tool_iterations"] == 2
    assert metrics["failed_iterations"] == 1
    assert metrics["self_correction_iterations"] == 1
    assert metrics["successful_iterations"] == 1
    assert metrics["early_closed_turns"] == 1
    assert metrics["tool_errors"] == 1


def test_tokens_and_times_are_derived():
    metrics = summarize(_recorder(), wall_seconds=36.0)

    assert metrics["est_input_tokens"] == 450
    assert metrics["est_output_tokens"] == 120
    assert metrics["est_output_tokens_failed_turns"] == 40
    assert metrics["est_tokens_after_first_failure"] == (150 + 60) + (200 + 20)
    assert metrics["model_seconds"] == pytest.approx(25.0)
    assert metrics["tool_seconds"] == pytest.approx(1.0 + 15.0)
    # Tool time inside model streaming: 4-5s and 15-20s.
    assert metrics["overlap_seconds"] == pytest.approx(1.0 + 5.0)
    assert metrics["first_tool_call_seconds"] == pytest.approx(4.0)
    assert metrics["est_tokens_per_iteration"] == pytest.approx(570 / 3)


def test_a_failed_call_marks_its_turn_failed():
    recorder = RunRecorder()
    recorder.turns = [TurnRecord(0, 0.0, end=1.0, made_tool_call=True)]
    recorder.tool_calls = [ToolCallRecord("f", 0.5, 0.9, ok=False, turn=0)]

    assert summarize(recorder, wall_seconds=1.0)["failed_iterations"] == 1


def test_native_tokens_and_seconds_per_iteration_are_derived():
    recorder = _recorder()
    recorder.turns[0].native_output = 50
    recorder.turns[2].native_output = 7

    metrics = summarize(recorder, wall_seconds=36.0)

    assert metrics["native_output_tokens"] == 57
    assert metrics["native_input_tokens"] is None
    assert metrics["seconds_per_iteration"] == pytest.approx(12.0)


def test_model_time_ends_at_the_last_token_not_when_the_stream_was_closed():
    # IPTC: generation ended at 4s, but the stream was only released at 10s,
    # once the code the tool call carried had finished running (5-9s).
    recorder = RunRecorder()
    recorder.turns = [
        TurnRecord(0, 0.0, last_token=4.0, end=10.0, completed=True),
    ]
    recorder.tool_calls = [ToolCallRecord("f", 5.0, 9.0, ok=True, turn=0)]

    metrics = summarize(recorder, wall_seconds=10.0)

    assert metrics["model_seconds"] == pytest.approx(4.0)
    assert metrics["overlap_seconds"] == pytest.approx(0.0)
