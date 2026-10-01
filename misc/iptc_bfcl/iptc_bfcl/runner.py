# Adapted from misc/iptc-parcs/iptc_parcs/runner.py.
import asyncio
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import langsmith
import openai
from jitgen_openai import IptcAgent, PtcAgent

from iptc_bfcl.baseline import ToolCallingAgent
from iptc_bfcl.config import RunSpec, Strategy
from iptc_bfcl.dataset import BfclEntry
from iptc_bfcl.grading import grade_calls
from iptc_bfcl.instrumentation import InstrumentedClient
from iptc_bfcl.metrics import RunRecorder, summarize
from iptc_bfcl.tools import BfclToolBridge

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.md").read_text(
    encoding="utf-8"
)
LANGSMITH_PROJECT = "iptc-bfcl"


@dataclass
class Outcome:
    """How one agent run ended, before grading.

    Attributes:
        answer: The model's final answer, empty if it gave none (always, for a
            single-response run).
        status: `finished` if the run ended as its mode intends, otherwise
            why it didn't.
        error: The run-ending exception, if any.
        wall_seconds: Time from the agent's start to its end.
    """

    answer: str = ""
    status: str = "agent_error"
    error: str = ""
    wall_seconds: float = 0.0


def system_prompt(entry: BfclEntry) -> str:
    """The harness's system prompt, followed by the entry's own, if any.

    Args:
        entry: The entry being run.

    Returns:
        The system message every arm sends for `entry`.
    """
    return f"{SYSTEM_PROMPT}\n\n{entry.system}" if entry.system else SYSTEM_PROMPT


def build_agent(
    spec: RunSpec, client: Any, bridge: BfclToolBridge, recorder: RunRecorder
) -> ToolCallingAgent | IptcAgent | PtcAgent:
    """Build the agent for `spec`'s arm; everything but the tool surface is shared.

    Args:
        spec: The run.
        client: The instrumented model client.
        bridge: The run's tool surface.
        recorder: Receives each tool result sent back to the model.

    Returns:
        The agent to run.
    """
    settings = spec.settings
    # A single-response run's second turn is answered by `_SingleResponseClient`.
    max_turns = settings.max_iterations if settings.full_cycle else 2
    options = {
        "seed": spec.seed,
        "temperature": settings.temperature,
        **spec.model.provider.routing_options(
            settings.reasoning_effort, settings.upstream
        ),
    }
    if spec.arm.strategy is Strategy.BASELINE:
        return ToolCallingAgent(
            client,
            spec.model.model,
            bridge,
            system_prompt=system_prompt(spec.entry),
            max_turns=max_turns,
            completion_options=options,
            on_tool_result=lambda result, failed: recorder.record_tool_result(
                result, execution_error=failed
            ),
        )
    agent_cls = IptcAgent if spec.arm.strategy is Strategy.IPTC else PtcAgent
    return agent_cls(
        client,
        spec.model.model,
        bridges=[bridge],  # type: ignore[list-item]  # duck-typed McpToolBridge
        system_prompt=system_prompt(spec.entry),
        language=spec.arm.language,  # type: ignore[arg-type]  # set for code arms
        max_turns=max_turns,
        executor_timeout=settings.executor_timeout,
        completion_options=options,
        on_tool_result=lambda result: recorder.record_tool_result(
            result, execution_error=result.startswith("execution error:")
        ),
    )


class _EmptyStream:
    """A model response with nothing in it, which agents take as an empty answer."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _SingleResponseClient:
    """Passes the first model call through and answers every later one empty.

    BFCL scores a single response. Every agent asks the model again once the
    first response's calls have finished; answering that request locally with
    an empty response ends the agent normally, with an empty final answer.
    Capping the agent at one turn would end it by raising instead, which
    marks every LangSmith trace as failed. The local answer is never sent to
    the model and isn't recorded as a turn.

    Args:
        inner: The (instrumented) client making the real first call.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._calls = 0
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> Any:
        self._calls += 1
        if self._calls > 1:
            return _EmptyStream()
        return await self._inner.chat.completions.create(**kwargs)


def _classify_exception(exc: BaseException) -> str:
    """Map a run-ending exception to a status; only real outages are infra."""
    text = str(exc)
    if isinstance(exc, TimeoutError):
        return "time_limit"
    if isinstance(exc, RuntimeError) and "no final answer" in text:
        return "iteration_limit"
    if "exceed_context_size" in text or "context size" in text:
        return "context_overflow"
    if isinstance(exc, openai.APIStatusError):
        return "infra_error" if exc.status_code >= 500 else "agent_error"
    if isinstance(exc, openai.APIConnectionError | ConnectionError):
        return "infra_error"
    if isinstance(exc, openai.APIError):
        # Errors inside a stream carry no status code, only the message.
        server_error = re.search(r"returned 5\d\d|\"code\":\s*5\d\d", text)
        return "infra_error" if server_error else "agent_error"
    return "agent_error"


async def run_agent(spec: RunSpec, client: Any, recorder: RunRecorder) -> Outcome:
    """Run `spec`'s agent on its entry against freshly simulated functions.

    Without `full_cycle`, the run ends once the first response's calls have
    finished, as BFCL scores a single response.

    Args:
        spec: The run.
        client: The instrumented model client.
        recorder: Receives the run's turns and tool calls.

    Returns:
        How the run ended.
    """
    outcome = Outcome()
    bridge = BfclToolBridge(spec.entry, recorder, spec.settings.tool_delay)
    if not spec.settings.full_cycle:
        client = _SingleResponseClient(client)
    agent = build_agent(spec, client, bridge, recorder)
    recorder.t0 = time.monotonic()
    try:
        try:
            with langsmith.tracing_context(
                project_name=LANGSMITH_PROJECT, metadata=spec.key()
            ):
                async with asyncio.timeout(spec.settings.time_limit):
                    outcome.answer = await agent.arun(spec.entry.user)
        finally:
            outcome.wall_seconds = recorder.now()
        outcome.status = "finished"
    except Exception as exc:  # noqa: BLE001  # every failure is a recorded status
        outcome.status = _classify_exception(exc)
        outcome.error = f"{type(exc).__name__}: {exc}"
    return outcome


def grade(outcome: Outcome, recorder: RunRecorder, entry: BfclEntry) -> dict[str, Any]:
    """Grade the run's calls and settle its status.

    `correct` judges the calls alone, whatever ended the run. The status of a
    finished run says whether those calls were right; otherwise it keeps the
    reason the run ended.

    Args:
        outcome: How the run ended.
        recorder: The run's turns and tool calls.
        entry: The entry the run answered.

    Returns:
        The grading columns of the run's CSV row, including the final status.
    """
    first_turn = next((t.index for t in recorder.turns if t.made_tool_call), None)
    graded = grade_calls(entry, recorder.tool_calls, first_turn)
    status = outcome.status
    if status == "finished":
        status = "answered_correct" if graded.correct else "answered_wrong"
        truncated = recorder.turns and recorder.turns[-1].finish_reason == "length"
        if truncated and not graded.correct:
            status = "output_truncated"
    return {
        "status": status,
        "correct": graded.correct,
        "first_turn_correct": graded.first_turn_correct,
        "grade_reason": graded.reason,
        "expected_calls": len(entry.ground_truth),
        "successful_calls": graded.successful_calls,
    }


def build_row(
    spec: RunSpec, started_at: str, outcome: Outcome, recorder: RunRecorder
) -> dict[str, Any]:
    """Assemble the run's `runs.csv` row.

    Args:
        spec: The run.
        started_at: When the run started, as an ISO timestamp.
        outcome: How the run ended.
        recorder: The run's turns and tool calls.

    Returns:
        The row, keyed by `results.RUN_FIELDS`.
    """
    return {
        **spec.key(),
        "attempt": spec.attempt,
        "seed": spec.seed,
        "order": spec.order,
        "started_at": started_at,
        **grade(outcome, recorder, spec.entry),
        "wall_seconds": outcome.wall_seconds,
        "error": outcome.error,
        "answer": outcome.answer[-2000:],
        **summarize(recorder, outcome.wall_seconds),
    }


async def execute_run(
    spec: RunSpec, *, prepare: bool
) -> tuple[dict[str, Any], RunRecorder]:
    """Run one agent end to end.

    Args:
        spec: The run.
        prepare: Bring the model to a fresh state first (an LM Studio reload).

    Returns:
        Its `runs.csv` row, and the recorder holding its turns and tool calls.
    """
    provider = spec.model.provider
    if prepare:
        await provider.prepare(spec.model.model, spec.settings.context_length)
    recorder = RunRecorder()
    client = InstrumentedClient(
        provider.build_client(),
        recorder,
        provider.request_options(),
        tool_result_limit=spec.settings.tool_result_limit,
    )
    started_at = datetime.now(UTC).isoformat()
    outcome = await run_agent(spec, client, recorder)
    await provider.fill_native_usage(recorder.turns)
    return build_row(spec, started_at, outcome, recorder), recorder
