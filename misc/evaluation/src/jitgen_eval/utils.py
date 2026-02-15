import asyncio
import time
import pandas as pd
from pydantic import BaseModel
import logging
import sys

from langchain_core.runnables import RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

from .prompt import TaskInput

import builtins
from datasets import Dataset


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that uses tqdm.write() to avoid interfering with progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)


async def timeout_async_iterator(aiter, timeout, task_id=None):
    """
    Wraps an asynchronous iterator 'aiter' so that each next item is awaited with a timeout.
    If a timeout occurs, the iteration is terminated.
    """
    task = None
    item_count = 0
    try:
        while True:
            try:
                # Create a task for the next item
                task = asyncio.create_task(aiter.__anext__())
                # Wait for the task with the specified timeout
                item = await asyncio.wait_for(task, timeout)
                task = None  # Task completed successfully
                item_count += 1
                yield item
            except asyncio.TimeoutError:
                tqdm.write(
                    f"⚠️  Timeout reached for task {task_id} after {item_count} items",
                    file=sys.stderr,
                )
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        tqdm.write(
                            f"❌ Error cancelling task for {task_id}: {e}",
                            file=sys.stderr,
                        )
                break
            except StopAsyncIteration:
                # The underlying async iterator is exhausted.
                break
    finally:
        # Ensure any pending task is cancelled on generator cleanup
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                tqdm.write(
                    f"❌ Error in finally cleanup for {task_id}: {e}", file=sys.stderr
                )


class TestCaseExecutionResult(BaseModel):
    success: bool
    has_timed_out: bool
    error: str | None = None
    output: str
    first_output_time: float
    total_time: float
    expected_output: str | None = None
    actual_output: str | None = None


def parse_test_case(test_case: str) -> str | None:
    """
    Parses a test case string into the source code and expected output.

    Args:
        test_case (str): The test case string containing source code and expected output.

    Returns:
        tuple[str, str | None]: A tuple containing the source code and the expected output.
    """
    parts = test_case.split("== ")
    return parts[1].strip() if len(parts) > 1 else None


async def execute_test_case_with_timeout(
    case_input: TaskInput,
    lcel_chain: RunnableSerializable[TaskInput, str],
    timeout: float = 30.0,
    task_id: str = "unknown",
) -> TestCaseExecutionResult:
    """
    Executes a test case with a timeout, capturing the output and timing information.

    Args:
        case_input (TaskInput): The input for the test case.
        lcel_chain (RunnableSerializable[dict, str]): The chain to execute.
        timeout (float): The maximum time to wait for the first output.
        task_id (str): Identifier for the task being executed.

    Returns:
        TestCaseExecutionResult: The result of the test case execution.
    """
    start_time = time.perf_counter()
    first_output_time: float | None = None
    has_timed_out = False
    output = ""

    try:
        async for text in timeout_async_iterator(
            lcel_chain.astream(case_input), timeout, task_id=task_id
        ):
            if first_output_time is None:
                first_output_time = time.perf_counter()

            if first_output_time - start_time > timeout:
                has_timed_out = True
                break

            output += text

        total_time = time.perf_counter() - start_time

        test_case = case_input["test_output"]

        return TestCaseExecutionResult(
            success=not has_timed_out,
            has_timed_out=has_timed_out,
            output=output,
            first_output_time=first_output_time - start_time
            if first_output_time
            else total_time,
            total_time=total_time,
            expected_output=test_case,
            actual_output=output if not has_timed_out else None,
        )
    except Exception as e:
        total_time = time.perf_counter() - start_time
        tqdm.write(
            f"❌ Exception in task {task_id}: {type(e).__name__}: {e}", file=sys.stderr
        )
        return TestCaseExecutionResult(
            success=False,
            has_timed_out="Timeout" in str(e),
            error=str(e),
            output="",
            first_output_time=first_output_time - start_time
            if first_output_time
            else total_time,
            total_time=total_time,
        )


async def execute_test_case(
    case_input: TaskInput, lcel_chain: RunnableSerializable[TaskInput, str]
) -> tuple[bool, str, float, float]:
    start_time = time.perf_counter()
    first_output: float | None = None
    try:
        result = ""
        async for text in lcel_chain.astream(case_input):
            if first_output is None:
                first_output = time.perf_counter()

            if first_output - start_time > 30:
                return (
                    True,
                    "Timeout",
                    time.perf_counter() - start_time,
                    first_output - start_time,
                )

            result += text
        return (
            False,
            result,
            time.perf_counter() - start_time,
            first_output - start_time,
        )
    except Exception as e:
        if first_output is None:
            first_output = time.perf_counter()
        end_time = time.perf_counter()
        return True, str(e), end_time - start_time, first_output - start_time


async def run_test_cases(
    llm: BaseChatModel,
    dataset: Dataset,
    lcel_chain: RunnableSerializable[TaskInput, str],
) -> pd.DataFrame:
    def disabled_input(*args, **kwargs):
        raise Exception("The input() function has been disabled.")

    # Override the built-in input
    old_input = builtins.input
    builtins.input = disabled_input

    tqdm.write("🔥 Warming up the model...", file=sys.stderr)
    await llm.ainvoke("hi")  # Warm up the model
    tqdm.write(
        f"✅ Model ready. Starting evaluation of {len(dataset)} test cases\n",
        file=sys.stderr,
    )

    results = []
    for idx, case in enumerate(tqdm(dataset, desc="Evaluating", unit="test")):
        task_id = case.get("task_id", f"case_{idx}")

        case_input: TaskInput = {
            "task": case["text"].replace("function", "Python code"),
            "example_test_input": case.get("example_test_input", ""),
            "example_test_output": case.get("example_test_output", ""),
            "test_input": case.get("test_input", ""),
            "test_output": case.get("test_output", ""),
        }

        # result = await execute_test_case(case_input, lcel_chain)
        result = await execute_test_case_with_timeout(
            case_input, lcel_chain, timeout=30, task_id=task_id
        )

        results.append((task_id, result))

    builtins.input = old_input
    tqdm.write(f"\n✅ All {len(dataset)} test cases completed", file=sys.stderr)

    df = pd.DataFrame(results, columns=["RowID", "ExecutionInfo"])
    df["ErrorOccurred"] = df["ExecutionInfo"].apply(lambda x: not x.success)
    df["ExecutionOutput"] = df["ExecutionInfo"].apply(lambda x: x.output)
    df["HasTimedOut"] = df["ExecutionInfo"].apply(lambda x: x.has_timed_out)
    df["ExecutionError"] = df["ExecutionInfo"].apply(lambda x: x.error)
    df["ExecutionTime"] = df["ExecutionInfo"].apply(lambda x: x.total_time)
    df["ExpectedOutput"] = df["ExecutionInfo"].apply(lambda x: x.expected_output)
    df["ActualOutput"] = df["ExecutionInfo"].apply(
        lambda x: x.actual_output.strip() if x.actual_output else None
    )
    df["CorrectOutput"] = df["ExecutionInfo"].apply(
        lambda x: (
            x.expected_output == x.actual_output.strip() if x.actual_output else False
        )
    )
    df["FirstOutput"] = df["ExecutionInfo"].apply(lambda x: x.first_output_time)
    df.drop(columns=["ExecutionInfo"], inplace=True)

    return df


def get_results_file_name(results_directory: str, model_name: str, dataset_name: str, chain_type: str) -> str:
    safe_model_name = model_name.replace("/", "__").replace(":", "_").replace(".", "_")
    return f"{results_directory}/{safe_model_name}_{chain_type}_results_{dataset_name}.csv"