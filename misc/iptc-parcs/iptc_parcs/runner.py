import asyncio
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import langsmith
import openai
from jitgen_openai import IptcAgent, McpToolBridge, PtcAgent

from iptc_parcs.baseline import ToolCallingAgent
from iptc_parcs.bridge import ExperimentBridge
from iptc_parcs.config import RunSpec, Strategy
from iptc_parcs.faults import FaultInjector
from iptc_parcs.instrumentation import InstrumentedClient
from iptc_parcs.metrics import RunRecorder, summarize
from iptc_parcs.task import TASK, extract_answer, reference, relative_error

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.md").read_text(
    encoding="utf-8"
)
LANGSMITH_PROJECT = "iptc-parcs"


@dataclass
class Outcome:
    """How one agent run ended, before grading.

    Attributes:
        answer: The model's final answer, empty if it gave none.
        status: `no_answer` if the agent returned, otherwise why it didn't.
        error: The run-ending exception, if any.
        wall_seconds: Time from the agent's start to its end.
    """

    answer: str = ""
    status: str = "agent_error"
    error: str = ""
    wall_seconds: float = 0.0


def build_agent(
    spec: RunSpec, client: Any, bridge: ExperimentBridge, recorder: RunRecorder
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
    options = {"seed": spec.seed, "temperature": spec.settings.temperature}
    if spec.arm.strategy is Strategy.BASELINE:
        return ToolCallingAgent(
            client,
            spec.model.model,
            bridge,
            system_prompt=SYSTEM_PROMPT,
            max_turns=spec.settings.max_iterations,
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
        system_prompt=SYSTEM_PROMPT,
        language=spec.arm.language,  # type: ignore[arg-type]  # set for code arms
        max_turns=spec.settings.max_iterations,
        executor_timeout=spec.settings.executor_timeout,
        completion_options=options,
        on_tool_result=lambda result: recorder.record_tool_result(
            result, execution_error=result.startswith("execution error:")
        ),
    )


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


async def run_agent(
    spec: RunSpec, inner: McpToolBridge, client: Any, recorder: RunRecorder
) -> Outcome:
    """Connect to PARCS through `inner` and run `spec`'s agent on the task.

    Args:
        spec: The run.
        inner: An unstarted bridge to PARCS; it is started and closed here.
        client: The instrumented model client.
        recorder: Receives the run's turns and tool calls.

    Returns:
        How the run ended. Failing before the agent started means PARCS was
        unreachable, which is an `infra_error`.
    """
    outcome = Outcome()
    started = False
    try:
        async with inner:
            bridge = ExperimentBridge(inner, recorder, FaultInjector(spec.scenario))
            agent = build_agent(spec, client, bridge, recorder)
            recorder.t0 = time.monotonic()
            started = True
            try:
                with langsmith.tracing_context(
                    project_name=LANGSMITH_PROJECT, metadata=spec.key()
                ):
                    async with asyncio.timeout(spec.settings.time_limit):
                        outcome.answer = await agent.arun(TASK)
            finally:
                outcome.wall_seconds = recorder.now()
            outcome.status = "no_answer"
    except Exception as exc:  # noqa: BLE001  # every failure is a recorded status
        outcome.status = _classify_exception(exc) if started else "infra_error"
        outcome.error = f"{type(exc).__name__}: {exc}"
    return outcome


def grade(outcome: Outcome, recorder: RunRecorder, tolerance: float) -> dict[str, Any]:
    """Check the answer against the exact VaR/CVaR and settle the run's status.

    Args:
        outcome: How the run ended.
        recorder: The run's turns and tool calls.
        tolerance: Relative error allowed in VaR and CVaR.

    Returns:
        The grading columns of the run's CSV row, including the final status.
    """
    status = outcome.status
    parsed = extract_answer(outcome.answer) if outcome.answer else None
    var_err = cvar_err = None
    correct = False
    if parsed is not None:
        ref = reference()
        var_err = relative_error(parsed[0], ref.var_99)
        cvar_err = relative_error(parsed[1], ref.cvar_99)
        correct = max(var_err, cvar_err) <= tolerance
        status = "answered_correct" if correct else "answered_wrong"
    elif status == "no_answer" and recorder.turns:
        if recorder.turns[-1].finish_reason == "length":
            status = "output_truncated"
    infra_incidents = sum(
        1 for c in recorder.tool_calls if c.error and "ConnectionError" in c.error
    )
    if infra_incidents and not correct:
        status = "infra_error"
    return {
        "status": status,
        "correct": correct,
        "var_99": parsed[0] if parsed else None,
        "cvar_99": parsed[1] if parsed else None,
        "var_rel_error": var_err,
        "cvar_rel_error": cvar_err,
        "infra_incidents": infra_incidents,
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
        **grade(outcome, recorder, spec.settings.tolerance),
        "wall_seconds": outcome.wall_seconds,
        "error": outcome.error,
        "answer": outcome.answer[-2000:],
        **summarize(recorder, outcome.wall_seconds),
    }


async def execute_run(spec: RunSpec) -> tuple[dict[str, Any], RunRecorder]:
    """Run one agent end to end against live PARCS.

    Args:
        spec: The run.

    Returns:
        Its `runs.csv` row, and the recorder holding its turns and tool calls.
    """
    provider = spec.model.provider
    await provider.prepare(spec.model.model, spec.settings.context_length)
    recorder = RunRecorder()
    client = InstrumentedClient(
        provider.build_client(),
        recorder,
        provider.request_options(),
        tool_result_limit=spec.settings.tool_result_limit,
    )
    started_at = datetime.now(UTC).isoformat()
    outcome = await run_agent(spec, McpToolBridge.sse(spec.parcs_url), client, recorder)
    await provider.fill_native_usage(recorder.turns)
    return build_row(spec, started_at, outcome, recorder), recorder
