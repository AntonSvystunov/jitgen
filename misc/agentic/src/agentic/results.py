import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from agentic.agents.base import ExecutionResult
from langchain_core.messages import AnyMessage


_CSV_COLUMNS = [
    "model_name",
    "mode",
    "task_id",
    "success",
    "is_timeout",
    "is_error",
    "final_result",
    "expected_answer",
    "is_correct",
    "steps_count",
    "total_time",
    "inference_time",
    "execution_time",
    "early_exit_on_error",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
]


def _sanitize(value: str) -> str:
    return re.sub(r"[^\w.-]", "_", value)


def make_run_stem(model_name: str, mode: str, dataset: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_sanitize(model_name)}__{mode}__{dataset}__{ts}"


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip().lower()


def _is_correct(output: str | None, expected: str | None) -> bool:
    return bool(output) and _normalize(output) == _normalize(expected)


def _message_to_dict(msg: AnyMessage) -> dict:
    return {
        "type": msg.__class__.__name__,
        "content": msg.content,
    }


def result_to_row(
    result: ExecutionResult,
    task_id: str,
    expected_answer: str | None,
    mode: str,
) -> dict:
    return {
        "model_name": result.model_name or "",
        "mode": mode,
        "task_id": task_id,
        "success": result.success,
        "is_timeout": result.timeout,
        "is_error": bool(result.error),
        "final_result": result.output or "",
        "expected_answer": expected_answer or "",
        "is_correct": _is_correct(result.output, expected_answer),
        "steps_count": result.steps_taken,
        "total_time": result.total_execution_time,
        "inference_time": result.inference_time,
        "execution_time": result.execution_time,
        "early_exit_on_error": result.early_exit_on_error,
        "total_tokens": result.total_tokens,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
    }


def write_jsonl_row(
    path: Path,
    result: ExecutionResult,
    task_id: str,
    expected_answer: str | None,
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = result_to_row(result, task_id, expected_answer, mode)
    row["messages"] = [_message_to_dict(m) for m in result.messages]
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
