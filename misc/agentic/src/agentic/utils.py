import asyncio
import logging
import sys
import time

import pandas as pd
from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from tqdm import tqdm

from .config import AgenticEvaluationConfig
from .prompt import TaskInput, format_user_message, CODEACT_SYSTEM_PROMPT
from .state import AgentState


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)


class TestCaseExecutionResult(BaseModel):
    success: bool
    has_timed_out: bool
    error: str | None = None
    output: str
    execution_turns: int
    total_time: float
    expected_output: str | None = None
    actual_output: str | None = None


def get_results_file_name(
    results_directory: str,
    model_name: str,
    dataset_name: str,
    chain_type: str,
) -> str:
    safe_model_name = (
        model_name.replace("/", "__").replace(":", "_").replace(".", "_")
    )
    return f"{results_directory}/{safe_model_name}_{chain_type}_results_{dataset_name}.csv"


async def execute_single_test_case(
    graph: CompiledStateGraph,
    task_input: TaskInput,
    config: AgenticEvaluationConfig,
    task_id: str = "unknown",
) -> TestCaseExecutionResult:
    """Run a single MBPP test case through the agent graph."""
    start_time = time.perf_counter()

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=CODEACT_SYSTEM_PROMPT),
            HumanMessage(content=format_user_message(task_input)),
        ],
        "task_input": dict(task_input),
        "execution_count": 0,
        "final_output": "",
        "has_error": False,
        "has_timed_out": False,
        "error_message": None,
    }

    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=config.test_case_timeout,
        )
        total_time = time.perf_counter() - start_time

        output = final_state.get("final_output", "")
        has_error = final_state.get("has_error", False)
        has_timed_out = final_state.get("has_timed_out", False)
        error_msg = final_state.get("error_message")
        exec_turns = final_state.get("execution_count", 0)

        return TestCaseExecutionResult(
            success=not has_error and not has_timed_out,
            has_timed_out=has_timed_out,
            error=error_msg,
            output=output,
            execution_turns=exec_turns,
            total_time=total_time,
            expected_output=task_input.get("test_output"),
            actual_output=output.strip() if output else None,
        )

    except asyncio.TimeoutError:
        total_time = time.perf_counter() - start_time
        tqdm.write(
            f"⚠️  Global timeout for task {task_id} after {total_time:.1f}s",
            file=sys.stderr,
        )
        return TestCaseExecutionResult(
            success=False,
            has_timed_out=True,
            error="Global timeout",
            output="",
            execution_turns=0,
            total_time=total_time,
            expected_output=task_input.get("test_output"),
            actual_output=None,
        )
    except Exception as exc:
        total_time = time.perf_counter() - start_time
        tqdm.write(
            f"❌ Exception in task {task_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return TestCaseExecutionResult(
            success=False,
            has_timed_out=False,
            error=str(exc),
            output="",
            execution_turns=0,
            total_time=total_time,
            expected_output=task_input.get("test_output"),
            actual_output=None,
        )


async def run_test_cases(
    graph: CompiledStateGraph,
    dataset: Dataset,
    config: AgenticEvaluationConfig,
    strategy_label: str,
) -> pd.DataFrame:
    """Iterate over *dataset* running each test case through *graph*."""
    import builtins

    def disabled_input(*args, **kwargs):
        raise Exception("The input() function has been disabled.")

    old_input = builtins.input
    builtins.input = disabled_input

    results: list[tuple[str, TestCaseExecutionResult]] = []

    for idx, case in enumerate(
        tqdm(dataset, desc=f"Evaluating ({strategy_label})", unit="test")
    ):
        task_id = case.get("task_id", f"case_{idx}")
        task_input: TaskInput = {
            "task": case["text"].replace("function", "Python code"),
            "example_test_input": case.get("example_test_input", ""),
            "example_test_output": case.get("example_test_output", ""),
            "test_input": case.get("test_input", ""),
            "test_output": case.get("test_output", ""),
        }

        result = await execute_single_test_case(
            graph, task_input, config, task_id=task_id
        )
        results.append((task_id, result))

    builtins.input = old_input
    tqdm.write(
        f"\n✅ All {len(dataset)} test cases completed ({strategy_label})",
        file=sys.stderr,
    )

    df = pd.DataFrame(results, columns=["RowID", "ExecutionInfo"])
    df["ErrorOccurred"] = df["ExecutionInfo"].apply(lambda x: not x.success)
    df["ExecutionOutput"] = df["ExecutionInfo"].apply(lambda x: x.output)
    df["HasTimedOut"] = df["ExecutionInfo"].apply(lambda x: x.has_timed_out)
    df["ExecutionError"] = df["ExecutionInfo"].apply(lambda x: x.error)
    df["ExecutionTime"] = df["ExecutionInfo"].apply(lambda x: x.total_time)
    df["ExecutionTurns"] = df["ExecutionInfo"].apply(lambda x: x.execution_turns)
    df["ExpectedOutput"] = df["ExecutionInfo"].apply(lambda x: x.expected_output)
    df["ActualOutput"] = df["ExecutionInfo"].apply(
        lambda x: x.actual_output.strip() if x.actual_output else None
    )
    df["CorrectOutput"] = df["ExecutionInfo"].apply(
        lambda x: (
            x.expected_output == x.actual_output.strip() if x.actual_output else False
        )
    )
    df.drop(columns=["ExecutionInfo"], inplace=True)

    return df
