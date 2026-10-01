from collections import Counter

import pytest

from iptc_parcs.main import _parse_args, main, plan_runs


def test_each_cell_runs_the_baseline_once_and_each_code_arm_per_language():
    runs = plan_runs(_parse_args(["--reps", "2"]))

    cells = Counter((spec.rep, spec.scenario) for spec in runs)
    assert set(cells.values()) == {5}
    assert len(runs) == 2 * 2 * 5
    first_cell = [spec for spec in runs if (spec.rep, spec.scenario) == (0, "natural")]
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


def test_languages_and_strategies_narrow_the_grid():
    args = _parse_args(
        [
            "--reps",
            "1",
            "--scenarios",
            "natural",
            "--strategies",
            "iptc",
            "--languages",
            "javascript",
        ]
    )

    [spec] = plan_runs(args)

    assert spec.arm.label == "iptc-javascript"
    assert spec.key()["language"] == "javascript"


def test_order_is_a_seeded_shuffle():
    first = [spec.arm.label for spec in plan_runs(_parse_args(["--reps", "1"]))]
    second = [spec.arm.label for spec in plan_runs(_parse_args(["--reps", "1"]))]

    assert first == second


def test_unknown_languages_are_rejected():
    with pytest.raises(SystemExit, match="unknown languages"):
        main(["--languages", "ruby", "--dry-run"])


def test_a_real_run_needs_a_parcs_url(monkeypatch):
    monkeypatch.delenv("PARCS_SERVER_URL", raising=False)
    monkeypatch.setattr("iptc_parcs.main.load_dotenv", lambda: None)

    with pytest.raises(SystemExit, match="PARCS_SERVER_URL"):
        main(["--reps", "1"])


def test_reasoning_efforts_are_a_grid_axis_in_the_run_key():
    args = _parse_args(
        [
            "--models",
            "openrouter:m",
            "--reps",
            "1",
            "--scenarios",
            "natural",
            "--reasoning-effort",
            "low,high",
            "--provider",
            "deepinfra",
        ]
    )

    runs = plan_runs(args)

    assert len(runs) == 2 * 5
    assert Counter(spec.key()["reasoning_effort"] for spec in runs) == {
        "low": 5,
        "high": 5,
    }
    assert {spec.key()["upstream"] for spec in runs} == {"deepinfra"}
    assert len({tuple(spec.key().values()) for spec in runs}) == len(runs)


def test_default_reasoning_effort_is_the_models_own():
    [spec, *_] = plan_runs(_parse_args(["--reps", "1"]))

    assert spec.settings.reasoning_effort == ""
    assert spec.key()["upstream"] == ""


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--reasoning-effort", "extreme"], "unknown reasoning efforts"),
        (["--reasoning-effort", "low"], "don't support --reasoning-effort"),
        (["--provider", "deepinfra"], "don't support --reasoning-effort"),
    ],
)
def test_invalid_routing_options_are_rejected_before_running(argv, message):
    with pytest.raises(SystemExit, match=message):
        main([*argv, "--dry-run"])  # the default model is an LM Studio one
