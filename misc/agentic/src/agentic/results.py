from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

TABLE_COLUMNS = [
    "model_name",
    "mode",
    "task_id",
    "success",
    "final_result",
    "expected_answer",
    "is_correct",
    "is_timeout",
    "is_error",
    "no_steps_left",
    "steps_count",
    "total_time",
    "steps_executed",
    "steps_duration",
]


@dataclass(slots=True)
class AgenticEvaluationRecord:
    model_name: str
    mode: str
    task_id: str
    success: bool
    final_result: str | None
    expected_answer: str
    is_correct: bool
    is_timeout: bool
    is_error: bool
    no_steps_left: bool
    steps_count: int
    total_time: float
    steps_executed: int
    steps_duration: list[float]
    messages: list[dict[str, str]]

    def to_table_row(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "task_id": self.task_id,
            "success": self.success,
            "final_result": self.final_result,
            "expected_answer": self.expected_answer,
            "is_correct": self.is_correct,
            "is_timeout": self.is_timeout,
            "is_error": self.is_error,
            "no_steps_left": self.no_steps_left,
            "steps_count": self.steps_count,
            "total_time": self.total_time,
            "steps_executed": self.steps_executed,
            "steps_duration": json.dumps(self.steps_duration),
        }

    def to_raw_record(self) -> dict[str, object]:
        return {
            **self.to_table_row(),
            "steps_duration": self.steps_duration,
            "messages": [message.copy() for message in self.messages],
        }


def sanitize_filename_component(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(".", "_")


def build_run_stem(
    model_name: str,
    mode: str,
    dataset: str,
    started_at: datetime,
) -> str:
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_filename_component(model_name)
    return f"{safe_model_name}_{mode}_{dataset}_{timestamp}"


def write_results(
    records: list[AgenticEvaluationRecord],
    *,
    results_directory: str,
    model_name: str,
    mode: str,
    dataset: str,
    started_at: datetime,
) -> tuple[Path, Path]:
    results_root = Path(results_directory)
    tables_directory = results_root / "tables"
    raw_directory = results_root / "raw"
    tables_directory.mkdir(parents=True, exist_ok=True)
    raw_directory.mkdir(parents=True, exist_ok=True)

    run_stem = build_run_stem(model_name, mode, dataset, started_at)
    csv_path = tables_directory / f"{run_stem}.csv"
    json_path = raw_directory / f"{run_stem}.json"

    pd.DataFrame(
        [record.to_table_row() for record in records],
        columns=TABLE_COLUMNS,
    ).to_csv(csv_path, index=False)

    json_path.write_text(
        json.dumps([record.to_raw_record() for record in records], indent=2),
        encoding="utf-8",
    )

    return csv_path, json_path