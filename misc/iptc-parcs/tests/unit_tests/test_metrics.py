import pytest

from iptc_parcs.metrics import RunRecorder, ToolCallRecord, TurnRecord, summarize


def _recorder() -> RunRecorder:
    recorder = RunRecorder()
    recorder.turns = [
        # Turn 0: streams 0-10s; its tool call fails (injected fault) at 4-5s.
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
        ToolCallRecord("create_session", 4.0, 5.0, ok=False, injected=True, turn=0),
        ToolCallRecord("create_session", 15.0, 16.0, ok=True, turn=1),
        ToolCallRecord(
            "run_layer",
            16.0,
            30.0,
            ok=True,
            total_elapsed_seconds=12.5,
            failure_count=0,
            turn=1,
        ),
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
    assert metrics["injected_faults"] == 1
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
    assert metrics["cluster_seconds"] == pytest.approx(12.5)
    assert metrics["first_tool_call_seconds"] == pytest.approx(4.0)
    assert metrics["est_tokens_per_iteration"] == pytest.approx(570 / 3)


def test_a_layer_with_failed_workers_marks_its_turn_failed():
    recorder = RunRecorder()
    recorder.turns = [TurnRecord(0, 0.0, end=1.0, made_tool_call=True)]
    recorder.tool_calls = [
        ToolCallRecord("run_layer", 0.5, 0.9, ok=True, failure_count=2, turn=0)
    ]

    assert summarize(recorder, wall_seconds=1.0)["failed_iterations"] == 1


def test_native_tokens_and_seconds_per_iteration_are_derived():
    recorder = _recorder()
    recorder.turns[0].native_output = 50
    recorder.turns[2].native_output = 7

    metrics = summarize(recorder, wall_seconds=36.0)

    assert metrics["native_output_tokens"] == 57
    assert metrics["native_input_tokens"] is None
    assert metrics["seconds_per_iteration"] == pytest.approx(12.0)
