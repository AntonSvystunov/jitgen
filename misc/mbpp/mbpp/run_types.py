from dataclasses import dataclass
from typing import Literal

from mbpp.results import Strategy

Status = Literal["ok", "timeout", "error"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The parameters that must be identical across both strategies' passes."""

    model: str
    seed: int
    temperature: float


@dataclass(slots=True)
class RunResult:
    """Outcome of running one dataset row under one strategy."""

    dataset_row: int
    case_index: int
    task_id: int
    strategy: Strategy
    config: RunConfig
    status: Status
    started_at: str
    elapsed_seconds: float
    first_statement_seconds: float | None
    completion_chunks: int
    api_prompt_tokens: int | None
    api_completion_tokens: int | None
    api_reasoning_tokens: int | None
    api_total_tokens: int | None
    early_exit: bool
    output: str
    actual_output: str
    expected_output: str
    correct_output: bool
    error_type: str | None
    error_detail: str | None
