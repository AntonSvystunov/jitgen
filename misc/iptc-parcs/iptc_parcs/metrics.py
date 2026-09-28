import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    """One model call (agent iteration). Times are seconds since run start.

    `est_*` token counts are a tiktoken `o200k_base` proxy computed the same
    way for every strategy (a stream closed early has no provider usage);
    `usage_*` are what the provider reported, when it did.
    """

    index: int
    request_start: float
    first_chunk: float | None = None
    first_reasoning: float | None = None
    first_content: float | None = None
    first_arguments: float | None = None
    end: float | None = None
    completed: bool = False
    finish_reason: str | None = None
    generation_id: str | None = None
    chunks: int = 0
    reasoning_text: str = ""
    content_text: str = ""
    arguments_text: str = ""
    usage_input: int | None = None
    usage_output: int | None = None
    usage_reasoning: int | None = None
    native_input: int | None = None
    native_output: int | None = None
    native_reasoning: int | None = None
    est_input: int = 0
    est_output: int = 0
    est_reasoning: int = 0
    truncated_tool_results: int = 0
    made_tool_call: bool = False
    tool_result: str | None = None
    execution_error: bool = False
    failed: bool = False


@dataclass
class ToolCallRecord:
    """One MCP tool call. Times are seconds since run start."""

    name: str
    start: float
    end: float
    ok: bool
    error: str | None = None
    injected: bool = False
    total_elapsed_seconds: float | None = None
    failure_count: int | None = None
    turn: int | None = None


@dataclass
class RunRecorder:
    """Collects the turns and tool calls of one agent run."""

    t0: float = field(default_factory=time.monotonic)
    turns: list[TurnRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def start_turn(self, est_input: int) -> TurnRecord:
        turn = TurnRecord(len(self.turns), self.now(), est_input=est_input)
        self.turns.append(turn)
        return turn

    def current_turn_index(self) -> int | None:
        return self.turns[-1].index if self.turns else None

    def record_tool_result(self, result: str, *, execution_error: bool) -> None:
        """Attach the tool result sent back to the model to the latest turn."""
        if not self.turns:
            return
        turn = self.turns[-1]
        turn.made_tool_call = True
        turn.tool_result = result
        turn.execution_error = turn.execution_error or execution_error


def finalize_turns(recorder: RunRecorder) -> None:
    """Mark each turn failed if its code raised or any of its tool calls failed."""
    for turn in recorder.turns:
        calls = [c for c in recorder.tool_calls if c.turn == turn.index]
        turn.failed = turn.execution_error or any(
            not c.ok or (c.failure_count or 0) > 0 for c in calls
        )


def _union_length(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    pieces = [
        (max(s1, s2), min(e1, e2))
        for s1, e1 in a
        for s2, e2 in b
        if min(e1, e2) > max(s1, s2)
    ]
    return _union_length(pieces)


def _sum(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def summarize(recorder: RunRecorder, wall_seconds: float) -> dict[str, Any]:
    """Derive the per-run metrics from the recorded turns and tool calls.

    Args:
        recorder: The run's turns and tool calls.
        wall_seconds: The run's total time.

    Returns:
        The metrics columns of the run's CSV row.
    """
    finalize_turns(recorder)
    turns, calls = recorder.turns, recorder.tool_calls
    tool_turns = [t for t in turns if t.made_tool_call]
    first_failed = next((t.index for t in turns if t.failed), None)
    after_failure = [
        t for t in turns if first_failed is not None and t.index > first_failed
    ]
    failed_turns = [t for t in turns if t.failed]

    model_spans = [(t.request_start, t.end) for t in turns if t.end is not None]
    tool_spans = [(c.start, c.end) for c in calls]
    iterations = len(turns)
    est_in = sum(t.est_input for t in turns)
    est_out = sum(t.est_output for t in turns)
    return {
        "iterations": iterations,
        "tool_iterations": len(tool_turns),
        "failed_iterations": len(failed_turns),
        "self_correction_iterations": sum(1 for t in after_failure if t.made_tool_call),
        "successful_iterations": sum(1 for t in tool_turns if not t.failed),
        "early_closed_turns": sum(
            1 for t in turns if t.end is not None and not t.completed
        ),
        "tool_calls": len(calls),
        "tool_errors": sum(1 for c in calls if not c.ok),
        "injected_faults": sum(1 for c in calls if c.injected),
        "truncated_tool_results": max(
            (t.truncated_tool_results for t in turns), default=0
        ),
        "est_input_tokens": est_in,
        "est_output_tokens": est_out,
        "est_reasoning_tokens": sum(t.est_reasoning for t in turns),
        "usage_input_tokens": _sum([t.usage_input for t in turns]),
        "usage_output_tokens": _sum([t.usage_output for t in turns]),
        "usage_reasoning_tokens": _sum([t.usage_reasoning for t in turns]),
        "native_input_tokens": _sum([t.native_input for t in turns]),
        "native_output_tokens": _sum([t.native_output for t in turns]),
        "native_reasoning_tokens": _sum([t.native_reasoning for t in turns]),
        "est_output_tokens_failed_turns": sum(t.est_output for t in failed_turns),
        "est_tokens_after_first_failure": sum(
            t.est_input + t.est_output for t in after_failure
        ),
        "model_seconds": _union_length(model_spans),
        "tool_seconds": _union_length(tool_spans),
        "overlap_seconds": _overlap(model_spans, tool_spans),
        "cluster_seconds": sum(
            c.total_elapsed_seconds or 0.0 for c in calls if c.name == "run_layer"
        ),
        "first_tool_call_seconds": min((c.start for c in calls), default=None),
        "seconds_per_iteration": wall_seconds / iterations if iterations else None,
        "est_tokens_per_iteration": (est_in + est_out) / iterations
        if iterations
        else None,
    }
