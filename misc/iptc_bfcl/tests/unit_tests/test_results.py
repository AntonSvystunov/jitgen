import pytest
from conftest import load_entry

from iptc_bfcl.config import Arm, RunSettings, RunSpec, Strategy, parse_model
from iptc_bfcl.metrics import ToolCallRecord
from iptc_bfcl.results import RUN_FIELDS, TOOL_CALL_FIELDS, ResultsWriter, run_key


def _spec(arm: Arm, delay: float = 0.1) -> RunSpec:
    return RunSpec(
        parse_model("lmstudio:m"),
        arm,
        load_entry("simple_0"),
        3,
        45,
        0,
        settings=RunSettings(tool_delay=delay),
    )


def test_written_runs_are_recognized_when_resuming(tmp_path):
    specs = [
        _spec(Arm(Strategy.BASELINE)),
        _spec(Arm(Strategy.IPTC, "javascript")),
        _spec(Arm(Strategy.IPTC, "javascript"), delay=1.0),
    ]
    writer = ResultsWriter(tmp_path)
    for spec in specs[:2]:
        writer.write({**spec.key(), "attempt": 1, "status": "finished"}, [], [])

    existing = {run_key(row) for row in ResultsWriter(tmp_path).existing_runs()}

    assert existing == {run_key(spec.key()) for spec in specs[:2]}
    # Another delay is another cell.
    assert run_key(specs[2].key()) not in existing


def test_tool_calls_are_written_with_their_arguments(tmp_path):
    spec = _spec(Arm(Strategy.BASELINE))
    writer = ResultsWriter(tmp_path)
    call = ToolCallRecord(
        "math.factorial", 0.0, 0.1, ok=True, arguments='{"number": 5}'
    )

    writer.write({**spec.key(), "attempt": 1}, [], [call])

    text = writer.tool_calls_path.read_text(encoding="utf-8")
    assert "arguments" in TOOL_CALL_FIELDS
    assert "math.factorial" in text
    assert '"{""number"": 5}"' in text


def test_run_fields_hold_the_key_and_the_grades():
    assert {
        "category",
        "entry_id",
        "tool_delay",
        "mode",
        "correct",
        "first_turn_correct",
        "grade_reason",
    } <= set(RUN_FIELDS)
    assert not {"scenario", "cluster_seconds", "injected_faults"} & set(RUN_FIELDS)


def test_a_results_dir_with_other_columns_is_refused_before_any_run(tmp_path):
    (tmp_path / "runs.csv").write_text("model,strategy,rep\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="use a new --out directory"):
        ResultsWriter(tmp_path)
