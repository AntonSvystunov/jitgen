import csv
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from iptc_parcs.metrics import ToolCallRecord, TurnRecord

RUN_KEY_FIELDS = ["model", "scenario", "strategy", "language", "rep"]

RUN_FIELDS = [
    *RUN_KEY_FIELDS,
    "attempt",
    "seed",
    "order",
    "started_at",
    "status",
    "correct",
    "var_99",
    "cvar_99",
    "var_rel_error",
    "cvar_rel_error",
    "wall_seconds",
    "iterations",
    "tool_iterations",
    "failed_iterations",
    "self_correction_iterations",
    "successful_iterations",
    "early_closed_turns",
    "tool_calls",
    "tool_errors",
    "injected_faults",
    "truncated_tool_results",
    "infra_incidents",
    "est_input_tokens",
    "est_output_tokens",
    "est_reasoning_tokens",
    "usage_input_tokens",
    "usage_output_tokens",
    "usage_reasoning_tokens",
    "native_input_tokens",
    "native_output_tokens",
    "native_reasoning_tokens",
    "est_output_tokens_failed_turns",
    "est_tokens_after_first_failure",
    "model_seconds",
    "tool_seconds",
    "overlap_seconds",
    "cluster_seconds",
    "first_tool_call_seconds",
    "seconds_per_iteration",
    "est_tokens_per_iteration",
    "error",
    "answer",
]

_TEXT_FIELDS = {"reasoning_text", "content_text", "arguments_text", "tool_result"}
TURN_FIELDS = (
    [*RUN_KEY_FIELDS, "attempt"]
    + [f.name for f in fields(TurnRecord) if f.name not in _TEXT_FIELDS]
    + ["tool_result_head"]
)
TOOL_CALL_FIELDS = [*RUN_KEY_FIELDS, "attempt"] + [
    f.name for f in fields(ToolCallRecord)
]


def run_key(row: dict[str, Any]) -> tuple[str, ...]:
    """The identity of one run in the grid, used for resuming.

    Args:
        row: A `RunSpec.key()`, or a row read back from `runs.csv`.

    Returns:
        The key fields as strings, so both sources compare equal.
    """
    return tuple(str(row[name]) for name in RUN_KEY_FIELDS)


class ResultsWriter:
    """Appends runs, turns and tool calls to CSV files in `directory`."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.runs_path = directory / "runs.csv"
        self.turns_path = directory / "turns.csv"
        self.tool_calls_path = directory / "tool_calls.csv"

    def existing_runs(self) -> list[dict[str, str]]:
        """Rows already in `runs.csv`."""
        if not self.runs_path.exists():
            return []
        with self.runs_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def write(
        self,
        run: dict[str, Any],
        turns: list[TurnRecord],
        tool_calls: list[ToolCallRecord],
    ) -> None:
        """Append one run with all of its turns and tool calls."""
        key = {name: run[name] for name in [*RUN_KEY_FIELDS, "attempt"]}
        _append(self.runs_path, RUN_FIELDS, [run])
        turn_rows = []
        for turn in turns:
            row = {k: v for k, v in asdict(turn).items() if k not in _TEXT_FIELDS}
            row["tool_result_head"] = (turn.tool_result or "")[:300]
            turn_rows.append({**key, **row})
        _append(self.turns_path, TURN_FIELDS, turn_rows)
        _append(
            self.tool_calls_path,
            TOOL_CALL_FIELDS,
            [{**key, **asdict(call)} for call in tool_calls],
        )


def _append(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
