from collections import Counter

import pytest
from conftest import FIXTURES, load_entry

from iptc_bfcl import main as main_module
from iptc_bfcl.main import _parse_args, load_selected_entries, main, plan_runs
from iptc_bfcl.metrics import RunRecorder
from iptc_bfcl.results import ResultsWriter

_ENTRIES = [load_entry("simple_0"), load_entry("parallel_0")]


def _plan(*argv: str):
    return plan_runs(_parse_args(list(argv)), _ENTRIES)


def test_each_cell_runs_the_baseline_once_and_each_code_arm_per_language():
    runs = _plan("--reps", "2", "--tool-delay", "0.1,1")

    cells = Counter(
        (spec.rep, spec.entry.id, spec.settings.tool_delay) for spec in runs
    )
    assert set(cells.values()) == {5}
    assert len(runs) == 2 * 2 * 2 * 5
    first_cell = [s for s in runs if (s.rep, s.entry.id) == (0, "simple_0")][:5]
    assert sorted(spec.arm.label for spec in first_cell) == [
        "baseline",
        "iptc-javascript",
        "iptc-python",
        "ptc-javascript",
        "ptc-python",
    ]
    assert sorted(spec.order for spec in first_cell) == [0, 1, 2, 3, 4]
    assert {spec.seed for spec in first_cell} == {42}
    assert len({tuple(spec.key().values()) for spec in runs}) == len(runs)
    assert {spec.key()["tool_delay"] for spec in runs} == {0.1, 1.0}


def test_languages_and_strategies_narrow_the_grid():
    runs = _plan("--strategies", "iptc", "--languages", "javascript")

    assert [spec.arm.label for spec in runs] == ["iptc-javascript"] * 2
    assert {spec.key()["category"] for spec in runs} == {"simple", "parallel"}


def test_order_is_a_seeded_shuffle():
    first = [spec.arm.label for spec in _plan()]
    second = [spec.arm.label for spec in _plan()]

    assert first == second


def test_entries_are_loaded_from_the_data_dir_and_sampled():
    args = _parse_args(
        ["--data-dir", str(FIXTURES), "--categories", "simple,parallel", "--limit", "1"]
    )

    entries = load_selected_entries(args)

    assert [e.category for e in entries] == ["simple", "parallel"]


def test_ids_pick_exact_entries():
    args = _parse_args(["--data-dir", str(FIXTURES), "--ids", "simple_89,parallel_0"])

    assert [e.id for e in load_selected_entries(args)] == ["simple_89", "parallel_0"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--languages", "ruby"], "unknown languages"),
        (["--categories", "irrelevance"], "unknown categories"),
        (["--strategies", "magic"], "unknown strategies"),
        (["--tool-delay", "-1"], "must not be negative"),
        (["--reasoning-effort", "extreme"], "unknown reasoning efforts"),
        (["--reasoning-effort", "low"], "don't support --reasoning-effort"),
        (["--provider", "deepinfra"], "don't support --reasoning-effort"),
    ],
)
def test_invalid_options_are_rejected_before_running(argv, message):
    with pytest.raises(SystemExit, match=message):
        main([*argv, "--data-dir", str(FIXTURES), "--dry-run"])


def test_dry_run_lists_the_runs(capsys):
    main(["--data-dir", str(FIXTURES), "--categories", "simple", "--dry-run"])

    assert capsys.readouterr().out.strip().endswith("10 runs")


async def test_runs_already_written_are_skipped_and_the_model_is_prepared_once(
    tmp_path, monkeypatch
):
    args = _parse_args(["--out", str(tmp_path), "--strategies", "baseline"])
    runs = plan_runs(args, _ENTRIES)
    ResultsWriter(tmp_path).write(
        {**runs[0].key(), "attempt": 1, "status": "answered_correct"}, [], []
    )
    executed = []

    async def fake_execute_run(spec, *, prepare):
        executed.append((spec.entry.id, prepare))
        return {
            **spec.key(),
            "attempt": 1,
            "status": "finished",
            "wall_seconds": 1.0,
            "iterations": 1,
        }, RunRecorder()

    monkeypatch.setattr(main_module, "execute_run", fake_execute_run)

    await main_module._run_all(args, runs)

    assert executed == [("parallel_0", True)]


def test_runs_stop_after_one_response_unless_full_cycle_is_asked_for():
    [single, *_] = _plan()
    [full, *_] = _plan("--full-cycle", "--max-iterations", "6")

    assert (single.settings.full_cycle, single.key()["mode"]) == (
        False,
        "single_response",
    )
    assert (full.settings.full_cycle, full.key()["mode"]) == (True, "full_cycle")
    assert full.settings.max_iterations == 6
