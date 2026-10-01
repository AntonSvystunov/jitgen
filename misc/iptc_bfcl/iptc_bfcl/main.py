# Adapted from misc/iptc-parcs/iptc_parcs/main.py.
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

from iptc_bfcl.config import (
    DEFAULT_MODELS,
    LANGUAGES,
    REASONING_EFFORTS,
    Arm,
    ModelSpec,
    RunSettings,
    RunSpec,
    Strategy,
    parse_model,
    plan_arms,
)
from iptc_bfcl.dataset import (
    CATEGORIES,
    DEFAULT_DATA_DIR,
    BfclEntry,
    ensure_downloaded,
    load_entries,
    select_entries,
)
from iptc_bfcl.metrics import RunRecorder
from iptc_bfcl.results import ResultsWriter, run_key
from iptc_bfcl.runner import execute_run
from iptc_bfcl.tools import BfclToolBridge

# Deliberately the same file iptc-parcs locks: both experiments drive the same
# local model server, and must never share it.
# FIXME: remove
LOCK_PATH = Path(tempfile.gettempdir()) / "iptc-parcs.lock"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _floats(value: str) -> list[float]:
    return [float(item) for item in _csv(value)]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="iptc-bfcl",
        description="Baseline vs PTC vs IPTC on BFCL with simulated async tools.",
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
    parser.add_argument(
        "--categories",
        type=_csv,
        default=list(CATEGORIES),
        help="comma-separated BFCL categories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="entries sampled per category; 0 for all",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--ids",
        type=lambda value: set(_csv(value)),
        default=None,
        help="comma-separated entry ids to run instead of a sample",
    )
    parser.add_argument(
        "--tool-delay",
        type=_floats,
        default=[0.1],
        help="comma-separated seconds every function call takes; a grid axis",
    )
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
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--tool-result-limit", type=int, default=20000)
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="run the whole agent loop until the model answers in text, instead "
        "of stopping after the first response's calls, as BFCL does",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="model round trips allowed per run with --full-cycle",
    )
    parser.add_argument("--executor-timeout", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--max-infra-retries", type=int, default=2)
    parser.add_argument(
        "--reload-each-run",
        action="store_true",
        help="reload a local model before every run, not only before its first",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
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
            f"another experiment holds {LOCK_PATH}; "
            "delete it if no experiment is running"
        )
        raise SystemExit(msg) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def load_selected_entries(args: argparse.Namespace) -> list[BfclEntry]:
    """Download (once) and sample the entries the experiment runs.

    Args:
        args: The parsed command line.

    Returns:
        The selected entries, category by category.

    Raises:
        SystemExit: If an entry's functions can't be served to every arm.
    """
    ensure_downloaded(args.categories, args.data_dir)
    entries = [
        entry
        for category in args.categories
        for entry in select_entries(
            load_entries(category, args.data_dir),
            limit=args.limit,
            sample_seed=args.sample_seed,
            ids=args.ids,
        )
    ]
    for entry in entries:
        try:
            BfclToolBridge(entry, RunRecorder(), delay=0.0)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    return entries


def plan_runs(args: argparse.Namespace, entries: list[BfclEntry]) -> list[RunSpec]:
    """Expand the CLI grid into runs, in execution order.

    Within each (model, reasoning effort, tool delay, rep, entry) cell every
    arm runs once with the same seed, in a seeded-shuffle order recorded per
    run, so no arm always goes first.

    Args:
        args: The parsed command line.
        entries: The entries to run.

    Returns:
        Every run of the grid.
    """
    base = RunSettings(
        temperature=args.temperature,
        time_limit=args.time_limit,
        full_cycle=args.full_cycle,
        max_iterations=args.max_iterations,
        executor_timeout=args.executor_timeout,
        context_length=args.context_length,
        tool_result_limit=args.tool_result_limit,
        upstream=tuple(args.provider),
        reload_each_run=args.reload_each_run,
    )
    arms = plan_arms(
        [Strategy(s) for s in args.strategies],
        [language for language in LANGUAGES if language in args.languages],
    )
    runs = []
    for model in [parse_model(m) for m in args.models]:
        for effort in args.reasoning_effort:
            for delay in args.tool_delay:
                settings = replace(base, reasoning_effort=effort, tool_delay=delay)
                runs.extend(_plan_cell_runs(args, model, settings, arms, entries))
    return runs


def _plan_cell_runs(
    args: argparse.Namespace,
    model: ModelSpec,
    settings: RunSettings,
    arms: list[Arm],
    entries: list[BfclEntry],
) -> list[RunSpec]:
    """Plan every rep and entry of one (model, reasoning effort, delay) triple.

    Args:
        args: The parsed command line.
        model: The model to run.
        settings: The run settings, with this triple's effort and delay.
        arms: The arms each cell runs.
        entries: The entries to run.

    Returns:
        The triple's runs, cell by cell.
    """
    runs = []
    for rep in range(args.reps):
        seed = args.seed_base + rep
        for entry in entries:
            order = list(arms)
            shuffle_key = (
                f"{seed}|{entry.id}|{model.label}|{settings.reasoning_effort}"
                f"|{settings.tool_delay}"
            )
            random.Random(shuffle_key).shuffle(order)
            runs.extend(
                RunSpec(model, arm, entry, rep, seed, position, settings=settings)
                for position, arm in enumerate(order)
            )
    return runs


def _label(spec: RunSpec, index: int, total: int) -> str:
    effort = spec.settings.reasoning_effort
    return (
        f"[{index}/{total}] {spec.model.label} {spec.arm.label} {spec.entry.id} "
        f"delay={spec.settings.tool_delay} {spec.key()['mode']} rep={spec.rep}"
        f"{f' effort={effort}' if effort else ''} attempt={spec.attempt}"
    )


def _validate(args: argparse.Namespace) -> None:
    """Reject options that would fail, or be silently ignored, mid-experiment.

    Args:
        args: The parsed command line.

    Raises:
        SystemExit: On an unknown language, category or reasoning effort, a
            negative delay, or a model that can't take the reasoning or
            provider options.
    """
    for kind, given, known in [
        ("languages", args.languages, LANGUAGES),
        ("categories", args.categories, CATEGORIES),
        ("strategies", args.strategies, [s.value for s in Strategy]),
    ]:
        unknown = sorted(set(given) - set(known))
        if unknown:
            msg = f"unknown {kind} {unknown}; known: {list(known)}"
            raise SystemExit(msg)
    unknown = sorted(set(args.reasoning_effort) - set(REASONING_EFFORTS))
    if unknown:
        known = [effort for effort in REASONING_EFFORTS if effort]
        msg = f"unknown reasoning efforts {unknown}; known: {known}"
        raise SystemExit(msg)
    if any(delay < 0 for delay in args.tool_delay):
        msg = "--tool-delay values must not be negative"
        raise SystemExit(msg)
    for model in [parse_model(m) for m in args.models]:
        for effort in args.reasoning_effort:
            try:
                model.provider.routing_options(effort, tuple(args.provider))
            except ValueError as exc:
                raise SystemExit(f"{model.label}: {exc}") from None


async def _run_all(args: argparse.Namespace, runs: list[RunSpec]) -> None:
    writer = ResultsWriter(args.out)
    existing = writer.existing_runs()
    done = {run_key(r) for r in existing if r["status"] != "infra_error"}
    infra_attempts = Counter(
        run_key(r) for r in existing if r["status"] == "infra_error"
    )
    prepared: set[str] = set()
    for index, spec in enumerate(runs, start=1):
        key = run_key(spec.key())
        if key in done:
            continue
        for attempt in range(infra_attempts[key] + 1, args.max_infra_retries + 2):
            spec = replace(spec, attempt=attempt)
            label = _label(spec, index, len(runs))
            print(f"{label} ...", flush=True)
            prepare = spec.settings.reload_each_run or spec.model.label not in prepared
            prepared.add(spec.model.label)
            row, recorder = await execute_run(spec, prepare=prepare)
            writer.write(row, recorder.turns, recorder.tool_calls)
            print(
                f"{label} -> {row['status']} in {row['wall_seconds']:.1f}s, "
                f"{row['iterations']} iterations",
                flush=True,
            )
            if row["status"] != "infra_error":
                break


def main(argv: list[str] | None = None) -> None:
    """Entry point for `iptc-bfcl`.

    Args:
        argv: Command-line arguments; `sys.argv` when `None`.
    """
    load_dotenv()
    args = _parse_args(argv)
    _validate(args)
    runs = plan_runs(args, load_selected_entries(args))
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
