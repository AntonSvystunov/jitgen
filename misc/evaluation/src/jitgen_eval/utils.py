import asyncio
import time
import pandas as pd
from pydantic import BaseModel

from langchain_core.runnables import RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

from .prompt import TaskInput

import builtins
from datasets import Dataset


async def timeout_async_iterator(aiter, timeout):
    """
    Wraps an asynchronous iterator 'aiter' so that each next item is awaited with a timeout.
    If a timeout occurs, the iteration is terminated.
    """
    while True:
        try:
            # Wait for the next item with the specified timeout.
            item = await asyncio.wait_for(aiter.__anext__(), timeout)
            yield item
        except asyncio.TimeoutError:
            print("Timeout reached while waiting for the next item.")
            break
        except StopAsyncIteration:
            # The underlying async iterator is exhausted.
            break


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
) -> TestCaseExecutionResult:
    """
    Executes a test case with a timeout, capturing the output and timing information.

    Args:
        case_input (TaskInput): The input for the test case.
        lcel_chain (RunnableSerializable[dict, str]): The chain to execute.
        timeout (float): The maximum time to wait for the first output.

    Returns:
        TestCaseExecutionResult: The result of the test case execution.
    """
    start_time = time.perf_counter()
    first_output_time: float | None = None
    has_timed_out = False
    output = ""

    try:
        async for text in timeout_async_iterator(
            lcel_chain.astream(case_input), timeout
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

    await llm.ainvoke("")  # Warm up the model

    results = []
    for case in tqdm(dataset):
        case_input: TaskInput = {
            "task": case["text"],
            "example_test_input": case.get("example_test_input", ""),
            "example_test_output": case.get("example_test_output", ""),
            "test_input": case.get("test_input", ""),
            "test_output": case.get("test_output", ""),            
        }

        # result = await execute_test_case(case_input, lcel_chain)
        result = await execute_test_case_with_timeout(
            case_input, lcel_chain, timeout=30
        )

        results.append((case["task_id"], result))

    builtins.input = old_input

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
        lambda x: x.expected_output == x.actual_output.strip()
        if x.actual_output
        else False
    )
    df["FirstOutput"] = df["ExecutionInfo"].apply(lambda x: x.first_output_time)
    df.drop(columns=["ExecutionInfo"], inplace=True)

    return df
