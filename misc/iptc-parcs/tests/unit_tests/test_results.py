import pytest

from iptc_parcs.config import Arm, RunSpec, Scenario, Strategy, parse_model
from iptc_parcs.results import RUN_FIELDS, ResultsWriter, run_key


def _spec(arm: Arm) -> RunSpec:
    return RunSpec(
        parse_model("lmstudio:m"), arm, Scenario.NATURAL, 3, 45, 0, "http://unused"
    )


def test_written_runs_are_recognized_when_resuming(tmp_path):
    specs = [_spec(Arm(Strategy.BASELINE)), _spec(Arm(Strategy.IPTC, "javascript"))]
    writer = ResultsWriter(tmp_path)
    for spec in specs:
        writer.write({**spec.key(), "attempt": 1, "status": "no_answer"}, [], [])

    existing = {run_key(row) for row in ResultsWriter(tmp_path).existing_runs()}

    assert existing == {run_key(spec.key()) for spec in specs}


def test_run_fields_hold_the_key_and_no_simulator_columns():
    assert {"language", "strategy", "scenario"} <= set(RUN_FIELDS)
    assert not any(
        name.startswith("sim_") or name in {"target", "cold_start"}
        for name in RUN_FIELDS
    )


def test_a_results_dir_with_other_columns_is_refused_before_any_run(tmp_path):
    (tmp_path / "runs.csv").write_text(
        "model,scenario,strategy,rep\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="use a new --out directory"):
        ResultsWriter(tmp_path)
