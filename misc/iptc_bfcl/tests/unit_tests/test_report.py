import math
from pathlib import Path

import pytest

from iptc_bfcl.metrics import ToolCallRecord, TurnRecord
from iptc_bfcl.report import (
    bootstrap_median_interval,
    load_results,
    main,
    paired_comparisons,
    sign_test,
    wilson_interval,
    write_report,
)
from iptc_bfcl.results import ResultsWriter

MODEL = "lmstudio:qwen/qwen3.6-27b"


def _write_run(
    writer: ResultsWriter,
    entry: str,
    strategy: str,
    language: str,
    *,
    wall: float,
    generation: float,
    correct: bool = True,
    model: str = MODEL,
) -> None:
    key = {
        "model": model,
        "category": "parallel",
        "entry_id": entry,
        "strategy": strategy,
        "language": language,
        "reasoning_effort": "",
        "upstream": "",
        "tool_delay": 0.3,
        "mode": "single_response",
        "rep": 0,
        "attempt": 1,
    }
    turn = TurnRecord(
        0, 0.0, first_arguments=1.0, last_arguments=2.0, last_token=generation, end=wall
    )
    calls = [
        ToolCallRecord("f", 1.5, 1.8, ok=True, turn=0),
        ToolCallRecord("f", 2.0, 2.3, ok=True, turn=0),
    ]
    writer.write(
        {
            **key,
            "status": "answered_correct" if correct else "answered_wrong",
            "correct": correct,
            "grade_reason": "" if correct else "simple_function_checker:wrong",
            "wall_seconds": wall,
            "model_seconds": generation,
            "tool_seconds": 0.6,
            "overlap_seconds": 0.3 if strategy == "iptc" else 0.0,
            "tool_calls": 2,
            "est_output_tokens": 100,
        },
        [turn],
        calls,
    )


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    writer = ResultsWriter(tmp_path)
    for i, entry in enumerate(["parallel_0", "parallel_1", "parallel_2"]):
        _write_run(writer, entry, "baseline", "", wall=3.0, generation=2.7)
        _write_run(writer, entry, "ptc", "python", wall=3.6 + i, generation=3.0 + i)
        _write_run(
            writer,
            entry,
            "iptc",
            "python",
            wall=3.1 + i,
            generation=3.0 + i,
            correct=i != 2,
        )
    _write_run(
        writer,
        "parallel_0",
        "ptc",
        "python",
        wall=9,
        generation=9,
        model="openrouter:x/throttled",
    )
    return tmp_path


def test_wilson_interval_brackets_the_proportion():
    low, high = wilson_interval(13, 16)
    assert low < 13 / 16 < high
    assert (round(low, 2), round(high, 2)) == (0.57, 0.93)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(16, 16)[1] == 1.0


def test_sign_test_is_exact_and_ignores_ties():
    assert sign_test([1.0] * 16) == pytest.approx(2 / 2**16)
    assert sign_test([1.0, -1.0]) == 1.0
    assert sign_test([0.0, 0.0]) == 1.0
    assert sign_test([1.0, 1.0, 1.0, 0.0]) == pytest.approx(0.25)


def test_bootstrap_interval_is_reproducible_and_contains_the_median():
    values = [0.1, 0.4, 0.6, 0.62, 0.65, 0.7, 1.2]
    first = bootstrap_median_interval(values)
    assert first == bootstrap_median_interval(values)
    assert first[0] <= 0.62 <= first[1]
    with pytest.raises(ValueError, match="no values"):
        bootstrap_median_interval([])


def test_results_load_with_excluded_models_left_out(results_dir: Path):
    results = load_results(results_dir, exclude=["throttled"])

    assert results.models == [MODEL]
    assert len(results.runs) == 9
    [ptc] = [r for r in results.by(MODEL, "ptc-python") if r.entry_id == "parallel_0"]
    assert ptc.execution == pytest.approx(0.6)
    assert ptc.argument_seconds == pytest.approx(1.0)


def test_iptc_is_compared_with_ptc_entry_by_entry(results_dir: Path):
    [comparison] = paired_comparisons(load_results(results_dir, ["throttled"]))

    assert (comparison.language, comparison.pairs) == ("python", 3)
    assert comparison.wall_median == pytest.approx(0.5)
    assert comparison.execution_median == pytest.approx(0.5)
    assert comparison.wall_faster == 3
    assert comparison.identical_outputs == 3
    assert math.isclose(comparison.wall_p, 0.25)


def test_the_report_writes_every_table_format(results_dir: Path):
    out = results_dir / "report"

    written = write_report(load_results(results_dir, ["throttled"]), out, figures=False)

    names = {path.name for path in written}
    for table in ("accuracy", "latency", "paired", "failures"):
        assert {f"{table}.md", f"{table}.tex", f"{table}.csv"} <= names
    accuracy = (out / "tables" / "accuracy.md").read_text(encoding="utf-8")
    assert "| qwen3.6-27b | 3/3 (100%)" in accuracy
    assert "2/3 (67%)" in accuracy
    latex = (out / "tables" / "paired.tex").read_text(encoding="utf-8")
    assert r"\toprule" in latex and r"\label{tab:paired}" in latex
    assert "parallel\\_2" in (out / "tables" / "failures.tex").read_text(
        encoding="utf-8"
    )


def test_figures_render(results_dir: Path):
    pytest.importorskip("matplotlib")
    out = results_dir / "report"

    written = write_report(load_results(results_dir, ["throttled"]), out)

    figures = {p.name for p in written if p.parent.name == "figures"}
    for name in (
        "timeline_example",
        "paired_savings",
        "overlap_mechanism",
        "time_breakdown",
        "accuracy",
    ):
        assert {f"{name}.pdf", f"{name}.png"} <= figures


def test_an_empty_results_dir_is_refused(tmp_path: Path):
    ResultsWriter(tmp_path)  # headers only
    for name in ("runs.csv", "turns.csv", "tool_calls.csv"):
        (tmp_path / name).write_text("model\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="no runs"):
        main([str(tmp_path), "--no-figures"])
