import argparse
import asyncio
import os
import random
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from iptc_parcs.config import (
    DEFAULT_MODELS,
    LANGUAGES,
    REASONING_EFFORTS,
    Arm,
    ModelSpec,
    RunSettings,
    RunSpec,
    Scenario,
    Strategy,
    parse_model,
    plan_arms,
)
from iptc_parcs.results import ResultsWriter, run_key
from iptc_parcs.runner import execute_run

LOCK_PATH = Path(tempfile.gettempdir()) / "iptc-parcs.lock"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="iptc-parcs",
        description="Baseline vs PTC vs IPTC on live PARCS (Monte Carlo VaR).",
    )
    parser.add_argument(
        "--models",
        type=_csv,
        default=DEFAULT_MODELS,
        help="comma-separated provider:model ids",
    )
    parser.add_argument("--strategies", type=_csv, default=list(Strategy))
    parser.add_argument(
        "--languages",
        type=_csv,
        default=list(LANGUAGES),
        help="languages the ptc/iptc arms write code in",
    )
    parser.add_argument("--scenarios", type=_csv, default=list(Scenario))
    parser.add_argument(
        "--reasoning-effort",
        type=lambda value: _csv(value) or [""],
        default=[""],
        help="comma-separated OpenRouter reasoning efforts to compare "
        f"({', '.join(e for e in REASONING_EFFORTS if e)}); "
        "the model's default when omitted",
    )
    parser.add_argument(
        "--provider",
        type=_csv,
        default=[],
        help="comma-separated OpenRouter provider slugs to pin every model to, "
        "in order, with no fallback",
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--tool-result-limit", type=int, default=20000)
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--executor-timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--max-infra-retries", type=int, default=2)
    parser.add_argument(
        "--parcs-url",
        default=os.environ.get("PARCS_SERVER_URL", ""),
        help="PARCS MCP SSE endpoint; defaults to PARCS_SERVER_URL",
    )
    parser.add_argument("--out", type=Path, default=Path("results") / "default")
    parser.add_argument("--dry-run", action="store_true", help="list the runs only")
    return parser.parse_args(argv)


@contextmanager
def _run_lock() -> Iterator[None]:
    """Stop two experiment processes from sharing the model server."""
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        msg = (
            f"another iptc-parcs run holds {LOCK_PATH}; "
            "delete it if no experiment is running"
        )
        raise SystemExit(msg) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def plan_runs(args: argparse.Namespace) -> list[RunSpec]:
    """Expand the CLI grid into runs, in execution order.

    Within each (model, reasoning effort, rep, scenario) cell every arm runs
    once with the same seed, in a seeded-shuffle order recorded per run: the live cluster keeps
    its worker pods warm between runs, so a fixed order would always hand the
    cold start to the same arm.

    Args:
        args: The parsed command line.

    Returns:
        Every run of the grid.
    """
    base = RunSettings(
        temperature=args.temperature,
        time_limit=args.time_limit,
        max_iterations=args.max_iterations,
        executor_timeout=args.executor_timeout,
        tolerance=args.tolerance,
        context_length=args.context_length,
        tool_result_limit=args.tool_result_limit,
        upstream=tuple(args.provider),
    )
    arms = plan_arms(
        [Strategy(s) for s in args.strategies],
        [language for language in LANGUAGES if language in args.languages],
    )
    runs = []
    for model in [parse_model(m) for m in args.models]:
        for effort in args.reasoning_effort:
            settings = replace(base, reasoning_effort=effort)
            runs.extend(_plan_cell_runs(args, model, settings, arms))
    return runs


def _plan_cell_runs(
    args: argparse.Namespace, model: ModelSpec, settings: RunSettings, arms: list[Arm]
) -> list[RunSpec]:
    """Plan every rep and scenario of one (model, reasoning effort) pair.

    Args:
        args: The parsed command line.
        model: The model to run.
        settings: The run settings, with this pair's reasoning effort.
        arms: The arms each cell runs.

    Returns:
        The pair's runs, cell by cell.
    """
    runs = []
    for rep in range(args.reps):
        seed = args.seed_base + rep
        for scenario in [Scenario(s) for s in args.scenarios]:
            order = list(arms)
            shuffle_key = f"{seed}|{scenario}|{model.label}|{settings.reasoning_effort}"
            random.Random(shuffle_key).shuffle(order)
            runs.extend(
                RunSpec(
                    model,
                    arm,
                    scenario,
                    rep,
                    seed,
                    position,
                    parcs_url=args.parcs_url,
                    settings=settings,
                )
                for position, arm in enumerate(order)
            )
    return runs


def _label(spec: RunSpec, index: int, total: int) -> str:
    effort = spec.settings.reasoning_effort
    return (
        f"[{index}/{total}] {spec.model.label} {spec.arm.label} "
        f"{spec.scenario} rep={spec.rep}"
        f"{f' effort={effort}' if effort else ''} attempt={spec.attempt}"
    )


def _validate(args: argparse.Namespace) -> None:
    """Reject options that would fail, or be silently ignored, mid-experiment.

    Args:
        args: The parsed command line.

    Raises:
        SystemExit: On an unknown language or reasoning effort, a model that
            can't take the reasoning or provider options, or a real run
            without a PARCS URL.
    """
    unknown = sorted(set(args.languages) - set(LANGUAGES))
    if unknown:
        msg = f"unknown languages {unknown}; known: {list(LANGUAGES)}"
        raise SystemExit(msg)
    unknown = sorted(set(args.reasoning_effort) - set(REASONING_EFFORTS))
    if unknown:
        known = [effort for effort in REASONING_EFFORTS if effort]
        msg = f"unknown reasoning efforts {unknown}; known: {known}"
        raise SystemExit(msg)
    for model in [parse_model(m) for m in args.models]:
        for effort in args.reasoning_effort:
            try:
                model.provider.routing_options(effort, tuple(args.provider))
            except ValueError as exc:
                raise SystemExit(f"{model.label}: {exc}") from None
    if not args.parcs_url and not args.dry_run:
        msg = "set PARCS_SERVER_URL (e.g. in .env) or pass --parcs-url"
        raise SystemExit(msg)


async def _run_all(args: argparse.Namespace, runs: list[RunSpec]) -> None:
    writer = ResultsWriter(args.out)
    existing = writer.existing_runs()
    done = {run_key(r) for r in existing if r["status"] != "infra_error"}
    infra_attempts = Counter(
        run_key(r) for r in existing if r["status"] == "infra_error"
    )
    for index, spec in enumerate(runs, start=1):
        key = run_key(spec.key())
        if key in done:
            continue
        for attempt in range(infra_attempts[key] + 1, args.max_infra_retries + 2):
            spec = replace(spec, attempt=attempt)
            label = _label(spec, index, len(runs))
            print(f"{label} ...", flush=True)
            row, recorder = await execute_run(spec)
            writer.write(row, recorder.turns, recorder.tool_calls)
            print(
                f"{label} -> {row['status']} in {row['wall_seconds']:.0f}s, "
                f"{row['iterations']} iterations",
                flush=True,
            )
            if row["status"] != "infra_error":
                break


def main(argv: list[str] | None = None) -> None:
    """Entry point for `iptc-parcs`.

    Args:
        argv: Command-line arguments; `sys.argv` when `None`.
    """
    load_dotenv()
    args = _parse_args(argv)
    _validate(args)
    runs = plan_runs(args)
    if args.dry_run:
        for spec in runs:
            print(run_key(spec.key()), f"seed={spec.seed} order={spec.order}")
        print(f"{len(runs)} runs")
        return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    with _run_lock():
        asyncio.run(_run_all(args, runs))


if __name__ == "__main__":
    main()
