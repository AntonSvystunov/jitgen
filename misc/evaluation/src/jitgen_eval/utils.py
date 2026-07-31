import asyncio
import time
import pandas as pd
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pydantic import BaseModel
import logging
import sys

from langchain_core.runnables import RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

from .chains import ChainBundle
from .config import config

from .prompt import TaskInput, task_prompt

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


@dataclass
class StreamStatus:
    """Out-param for :func:`timeout_async_iterator`.

    The iterator swallows the timeout so the caller keeps whatever output it has
    already accumulated; this records that a timeout happened so the caller does
    not report the truncated run as a success.
    """

    timed_out: bool = False


async def timeout_async_iterator(aiter, timeout, task_id=None, status=None):
    """
    Wraps an asynchronous iterator 'aiter' so that each next item is awaited with a timeout.
    If a timeout occurs, the iteration is terminated and ``status.timed_out`` is set.
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
                if status is not None:
                    status.timed_out = True
                tqdm.write(
                    f"⚠️  Timeout reached for task {task_id} after {item_count} items",
                    file=sys.stderr,
                )
                logger.warning(
                    "Timeout for task %s after %d items (%.1fs per-chunk limit)",
                    task_id,
                    item_count,
                    timeout,
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


_JITGEN_ERROR_MARKER = "[JITGen error:"


def _extract_inline_error(output: str) -> str | None:
    """Pull the message out of a ``[JITGen error: ...]`` chunk, if present."""
    start = output.find(_JITGEN_ERROR_MARKER)
    if start == -1:
        return None
    start += len(_JITGEN_ERROR_MARKER)
    end = output.find("]", start)
    return output[start : end if end != -1 else len(output)].strip()


def _looks_like_timeout(message: str) -> bool:
    """Whether *message* describes a timeout.

    Both arms surface timeouts only as text by the time they reach here, and the
    wording differs between them, so match on either spelling rather than one
    exact substring.
    """
    lowered = message.lower()
    return "timed out" in lowered or "timeout" in lowered


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

    status = StreamStatus()

    try:
        async for text in timeout_async_iterator(
            lcel_chain.astream(case_input), timeout, task_id=task_id, status=status
        ):
            if first_output_time is None:
                first_output_time = time.perf_counter()

            if first_output_time - start_time > timeout:
                has_timed_out = True
                break

            output += text

        total_time = time.perf_counter() - start_time

        # A per-chunk timeout truncates the stream; don't report that as success.
        has_timed_out = has_timed_out or status.timed_out

        # The incremental parser reports execution failures *in band*, as a
        # "[JITGen error: ...]" chunk, instead of raising the way the sequential
        # chain does.  Without picking that up here, a run whose code timed out
        # or blew up still lands in the results as a clean success — and the two
        # arms are not comparable on ErrorOccurred or HasTimedOut.
        inline_error = _extract_inline_error(output)
        if inline_error is not None:
            has_timed_out = has_timed_out or _looks_like_timeout(inline_error)

        test_case = case_input["test_output"]
        failed = has_timed_out or inline_error is not None

        return TestCaseExecutionResult(
            success=not failed,
            has_timed_out=has_timed_out,
            error=inline_error,
            output=output,
            first_output_time=first_output_time - start_time
            if first_output_time
            else total_time,
            total_time=total_time,
            expected_output=test_case,
            actual_output=output if not failed else None,
        )
    except Exception as e:
        total_time = time.perf_counter() - start_time
        tqdm.write(
            f"❌ Exception in task {task_id}: {type(e).__name__}: {e}", file=sys.stderr
        )
        # A failing test case is an expected outcome (the model wrote bad code),
        # so keep this concise; the full traceback is available at DEBUG.
        logger.warning("Task %s failed: %s: %s", task_id, type(e).__name__, e)
        logger.debug("Traceback for task %s", task_id, exc_info=True)
        return TestCaseExecutionResult(
            success=False,
            has_timed_out=_looks_like_timeout(str(e)),
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


def _build_case_input(case: dict) -> TaskInput:
    return {
        "task": case["text"].replace("function", "Python code"),
        "example_test_input": case.get("example_test_input", ""),
        "example_test_output": case.get("example_test_output", ""),
        "test_input": case.get("test_input", ""),
        "test_output": case.get("test_output", ""),
    }


async def _warm_up(llm: BaseChatModel, case_input: TaskInput | None) -> None:
    """Load the model and prime its KV cache with a representative prompt.

    Warming with a bare ``"hi"`` shares no prefix with the evaluation prompts, so
    the first measured case absorbs a cold prefill of the long system prompt
    (~82ms measured: 112ms time-to-first-token against 30ms once warm) that no
    later case pays.  Warming on a real prompt moves that cost out of the
    measured region instead of charging it to whichever arm happens to run first.
    """
    tqdm.write("🔥 Warming up the model...", file=sys.stderr)
    if case_input is None:
        await llm.ainvoke("hi")
    else:
        await (task_prompt | llm).ainvoke(case_input)


def _to_dataframe(rows: list[tuple[str, TestCaseExecutionResult]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["RowID", "ExecutionInfo"])
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


async def run_test_cases(
    llm: BaseChatModel,
    dataset: Dataset,
    chain_factories: Mapping[str, Callable[[], ChainBundle]],
) -> dict[str, pd.DataFrame]:
    """Run every test case through every arm under matched conditions.

    Each factory is called once per test case, so no REPL namespace, JITGen
    session or marker-stripper state leaks between cases.  Sharing a single chain
    would also let a cancelled stream (see :func:`timeout_async_iterator`)
    corrupt the session used by subsequent cases.

    Each arm gets its own full pass over the dataset, and every pass is preceded
    by an identical warm-up.  That combination is what puts the arms on equal
    footing, and it is subtler than it looks.

    Now that the model stays resident between requests (see
    ``EvaluationConfig.ollama_keep_alive``) its KV cache survives, and **the text
    the model generates depends on the cache state the previous request left
    behind** — measured directly: the same prompt, at temperature 0 with a fixed
    seed, produces different completions depending on which request preceded it.
    Determinism therefore requires every arm to see the same *sequence* of
    predecessors, which a full pass per arm gives for free: case *i* follows case
    *i-1* in every pass.  Both arms then execute the identical generated program
    and the only variable left is the execution strategy.

    Interleaving the arms case by case looks fairer — it would balance the
    cheaper prefill that the second arm gets against a repeated prompt — but it
    destroys exactly that property: an arm's call would be preceded by the *other*
    arm's identical prompt, the completions diverge, and the arms end up being
    scored on different programs.  Measured, that cost one arm a correctness
    point outright.  A ~1% timing tailwind is the cheaper problem, and warming up
    before each pass removes most of it by making both arms start from the same
    warm state instead of charging the first arm for a cold prefill.

    Returns one DataFrame per arm, keyed by the names given in *chain_factories*.
    """
    def disabled_input(*args, **kwargs):
        raise Exception("The input() function has been disabled.")

    arm_names = list(chain_factories)
    results: dict[str, list[tuple[str, TestCaseExecutionResult]]] = {}
    cases = [
        (case.get("task_id", f"case_{idx}"), _build_case_input(case))
        for idx, case in enumerate(dataset)
    ]

    # Override the built-in input
    old_input = builtins.input
    builtins.input = disabled_input
    try:
        for name in arm_names:
            # One identical warm-up per pass, so no arm is charged for the cold
            # prefill of the long system prompt and every pass starts from the
            # same cache state.
            await _warm_up(llm, cases[0][1] if cases else None)
            tqdm.write(
                f"✅ Model ready. Evaluating {len(cases)} test cases — {name}\n",
                file=sys.stderr,
            )

            rows: list[tuple[str, TestCaseExecutionResult]] = []
            for task_id, case_input in tqdm(cases, desc=f"Evaluating {name}", unit="test"):
                bundle = chain_factories[name]()
                try:
                    result = await execute_test_case_with_timeout(
                        case_input,
                        bundle.chain,
                        timeout=config.test_case_timeout,
                        task_id=task_id,
                    )
                finally:
                    await bundle.aclose()

                if result.success and not result.output and result.error is None:
                    logger.warning(
                        "Task %s (%s) produced no output and no error — the chain "
                        "yielded zero chunks in %.3fs",
                        task_id,
                        name,
                        result.total_time,
                    )

                rows.append((task_id, result))
            results[name] = rows
    finally:
        builtins.input = old_input

    tqdm.write(f"\n✅ All {len(cases)} test cases completed", file=sys.stderr)

    return {name: _to_dataframe(rows) for name, rows in results.items()}


def get_results_file_name(
    results_directory: str, model_name: str, dataset_name: str, chain_type: str
) -> str:
    safe_model_name = model_name.replace("/", "__").replace(":", "_").replace(".", "_")
    return (
        f"{results_directory}/{safe_model_name}_{chain_type}_results_{dataset_name}.csv"
    )
