from __future__ import annotations

import csv
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from mbpp.models import ModelSpec

if TYPE_CHECKING:
    # Only for type hints: `run_types.py` imports Strategy from *here*, so a
    # real (non-TYPE_CHECKING) import of RunResult would form a cycle.
    from mbpp.run_types import RunResult


# Lives here, not in main.py: `assert_unique_csv_stems` needs to enumerate
# strategy values at runtime, and main.py imports from this module, so
# Strategy has to be defined somewhere with no dependency on main.py.
class Strategy(StrEnum):
    INCREMENTAL = "incremental"
    SEQUENTIAL = "sequential"


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CSV_FIELDNAMES = [
    "Model",
    "ModelLabel",
    "Provider",
    "Strategy",
    "TaskId",
    "CaseIndex",
    "DatasetRow",
    "Seed",
    "StartedAt",
    "Outcome",
    "ErrorOccurred",
    "ErrorType",
    "ExecutionError",
    "HasTimedOut",
    "EarlyExit",
    "ExecutionTime",
    "FirstExecutedStatement",
    "ExecutionOutput",
    "ActualOutput",
    "ExpectedOutput",
    "CorrectOutput",
    "InputTokens",
    "OutputTokens",
    "ReasoningTokens",
    "TotalTokens",
    "ClientCompletionChunks",
]

# `ClientCompletionChunks` is the one column beyond the literally-specified
# schema: a count of non-empty stream delta events (as opposed to
# OutputTokens, the provider-reported token figure) -- one event can carry
# more than one token, so this is a chunk count, not a token count. It's
# what actually survives an incremental early-exit, since the provider's own
# final usage chunk never arrives when the stream is aborted mid-block --
# the metric H2 depends on.

_OUTCOME_LABELS = {"ok": "ok", "timeout": "stalled", "error": "error"}


def sanitize_filename_component(value: str) -> str:
    """Make a model identifier safe to use as (part of) a filename.

    `/` -> `__` (matches the given example
    `google__gemma-4-26b-a4b__incremental.csv`); `:` -> `-` since Windows
    forbids `:` in filenames and Ollama ids commonly contain one (e.g.
    `xingyaow/codeact-agent-mistral:latest`). `__` stays reserved as the
    field separator between the model and the strategy suffix.

    Args:
        value: A raw model identifier.

    Returns:
        A filesystem-safe version of `value`.
    """
    return value.replace("/", "__").replace(":", "-")


def csv_path_for(spec: ModelSpec, strategy: Strategy) -> Path:
    """Build the result CSV path for one (model, strategy) pass.

    Args:
        spec: The model the CSV belongs to.
        strategy: The strategy pass the CSV belongs to.

    Returns:
        `misc/mbpp/results/{sanitized model}__{strategy}.csv`.
    """
    stem = sanitize_filename_component(spec.model)
    return RESULTS_DIR / f"{stem}__{strategy.value}.csv"


def assert_unique_csv_stems(models: list[ModelSpec]) -> None:
    """Fail loudly before any run starts if two models would collide on a CSV path.

    Args:
        models: The registry to check.

    Raises:
        RuntimeError: Two `ModelSpec`s sanitize to the same (model, strategy) filename.
    """
    seen: dict[Path, ModelSpec] = {}
    for spec, strategy in product(models, Strategy):
        path = csv_path_for(spec, strategy)
        collision = seen.get(path)
        if collision is not None:
            msg = (
                f"models {collision.model!r} and {spec.model!r} both sanitize "
                f"to {path.name!r} -- rename one to avoid overwriting the other's results"
            )
            raise RuntimeError(msg)
        seen[path] = spec


def to_csv_row(result: RunResult, spec: ModelSpec) -> dict[str, object]:
    """Map one `RunResult` to a `CSV_FIELDNAMES`-shaped row.

    Args:
        result: The case outcome to serialize.
        spec: The model the result was produced under.

    Returns:
        A dict keyed by `CSV_FIELDNAMES`, with `None` values already
        coerced to `""` (rather than relying on `csv.DictWriter` to do it).
    """
    return {
        "Model": spec.model,
        "ModelLabel": spec.label,
        "Provider": spec.provider.name,
        "Strategy": result.strategy.value,
        "TaskId": result.task_id,
        "CaseIndex": result.case_index,
        "DatasetRow": result.dataset_row,
        "Seed": result.config.seed,
        "StartedAt": result.started_at,
        "Outcome": _OUTCOME_LABELS[result.status],
        "ErrorOccurred": result.status == "error",
        "ErrorType": result.error_type or "",
        "ExecutionError": result.error_detail or "",
        "HasTimedOut": result.status == "timeout",
        "EarlyExit": result.early_exit,
        "ExecutionTime": result.elapsed_seconds,
        "FirstExecutedStatement": (
            result.first_statement_seconds
            if result.first_statement_seconds is not None
            else ""
        ),
        "ExecutionOutput": result.output,
        "ActualOutput": result.actual_output,
        "ExpectedOutput": result.expected_output,
        "CorrectOutput": result.correct_output,
        "InputTokens": (
            result.api_prompt_tokens if result.api_prompt_tokens is not None else ""
        ),
        "OutputTokens": (
            result.api_completion_tokens
            if result.api_completion_tokens is not None
            else ""
        ),
        "ReasoningTokens": (
            result.api_reasoning_tokens
            if result.api_reasoning_tokens is not None
            else ""
        ),
        "TotalTokens": (
            result.api_total_tokens if result.api_total_tokens is not None else ""
        ),
        "ClientCompletionChunks": result.completion_chunks,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write `rows` to `path` as a `CSV_FIELDNAMES`-header CSV, overwriting any existing file.

    Args:
        path: Destination file, created (with parents) if it doesn't exist.
        rows: Rows shaped like `to_csv_row`'s output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="": otherwise the csv module's own line terminator combines
    # with Windows' text-mode translation to double every CR. utf-8: model
    # output can contain non-ASCII text, which would otherwise
    # UnicodeEncodeError under the default cp1252 codepage on Windows.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
