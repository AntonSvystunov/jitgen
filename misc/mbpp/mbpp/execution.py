import asyncio
import re
import time
from datetime import UTC, datetime

from jitgen import (
    ExecutionError,
    InProcPythonExecutor,
    JitGenError,
    Session,
    StreamDriver,
    create_python_session,
)
from jitgen.segmenters import markdown_code
from openai import AsyncOpenAI, omit

from mbpp.dataset import IndexedCase
from mbpp.results import Strategy
from mbpp.run_types import RunConfig, RunResult, Status

CASE_TIMEOUT_SECONDS = 60.0

# Matches `jitgen.segmenters.marker.markdown_code("python")`'s own start/end
# marker pair -- kept as plain strings, not a regex, since `_extract_code_block`
# needs the same substring-search semantics `MarkerSegmenter` uses, not
# pattern matching.
_CODE_FENCE_START = "```python"
_CODE_FENCE_END = "```"

# Pulls the leading class name off jitgen's own `"ClassName: message"`
# error formatting (`jitgen.executors.python._describe`,
# `jitgen.session._describe`).
_ERROR_TYPE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")


def _error_type(message: str, fallback: str) -> str:
    """Pull the exception class name off the front of a `"ClassName: message"` string.

    Args:
        message: A formatted error message, as produced by jitgen's own
            `_describe()` helper.
        fallback: Used when `message` doesn't have that shape — e.g. an
            `ExtractionError`'s message is a raw parser error, not a
            rendered Python exception.

    Returns:
        The leading class name, or `fallback`.
    """
    match = _ERROR_TYPE_RE.match(message)
    return match.group(1) if match else fallback


def _actual_output(output: str) -> str:
    """Trim captured stdout down to just the answer.

    The system prompt (`mbpp/prompts.py`) mandates the response end with
    exactly one `print(repr(answer))`, so the last non-empty line is
    normally the entire answer even if earlier debug prints exist.

    Args:
        output: The full captured stdout of a successful case.

    Returns:
        The last non-empty line, stripped; `""` if `output` has none.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


def _extract_code_block(raw: str) -> str:
    """Pull every fenced ```python code block out of a full model response.

    Deliberately independent of jitgen's `MarkerSegmenter`: the sequential
    baseline is meant to be a plain generate-then-execute pipeline with none
    of jitgen's own extraction machinery in the loop, so it gets its own
    minimal string-based extractor instead of reusing `markdown_code`. Its
    *matching semantics* still have to agree with `MarkerSegmenter`'s,
    though: a plain substring search for the opening/closing marker, re-armed
    after every closing marker so a response with more than one fenced block
    has every block's contents concatenated, not just the first. Without
    this a response with e.g. a "here's the function" block followed by a
    separate "here's a call to it" block would silently execute less code
    under sequential than incremental ran from the identical response.

    A missing closing fence on the last (or only) block is not an error —
    the model may have been cut off mid-block, or omitted it entirely —
    everything from that opening fence to the end of the response is used
    instead. No opening fence anywhere in the response is still an error,
    since then there is no code to run at all.

    Args:
        raw: The full, already-collected model response.

    Returns:
        Every fenced block's contents, concatenated in order.

    Raises:
        ValueError: `raw` contains no ` ```python ` opening fence anywhere.
    """
    blocks: list[str] = []
    pos = 0
    while True:
        open_idx = raw.find(_CODE_FENCE_START, pos)
        if open_idx == -1:
            break
        content_start = open_idx + len(_CODE_FENCE_START)
        close_idx = raw.find(_CODE_FENCE_END, content_start)
        if close_idx == -1:
            blocks.append(raw[content_start:])
            break
        blocks.append(raw[content_start:close_idx])
        pos = close_idx + len(_CODE_FENCE_END)

    if not blocks:
        msg = "no ```python code block found in response"
        raise ValueError(msg)
    return "".join(blocks)


async def run_case(
    client: AsyncOpenAI,
    config: RunConfig,
    case: IndexedCase,
    messages: list[dict[str, str]],
    strategy: Strategy,
    stream_options: dict[str, bool] | None,
) -> RunResult:
    """Run one dataset row under one strategy, timed and classified uniformly.

    Incremental dispatches each statement through jitgen's grammar-aware
    `Session`/`StreamDriver` as its boundary fixes. Sequential is a plain
    generate-then-execute baseline with none of that machinery: it collects
    the full response, pulls the code block out with a regex
    (`_extract_code_block`), and hands the whole block to a fresh
    `InProcPythonExecutor` in a single `aexecute` call. Execution failures
    from either path are classified the same way, since sequential wraps its
    outcome in the same `ExecutionError` type the session itself raises —
    timing, token counting, and success/timeout/error classification are a
    single shared tail so neither strategy defines its own metrics.

    Args:
        client: The chat completions client to stream from.
        config: The pinned run configuration (model, seed, temperature).
        case: The dataset row being solved, with its stable identity.
        messages: `case` rendered to chat messages, identical for both strategies.
        strategy: Which execution strategy to measure.
        stream_options: Provider-specific `stream_options` for the request
            (e.g. `{"include_usage": True}`), or `None` for a provider that
            rejects the parameter or doesn't need it.

    Returns:
        The classified outcome, always returned rather than raised, so a
        failing case doesn't abort the rest of the pass.
    """
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    output = ""
    completion_chunks = 0
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }
    early_exit = False
    status: Status = "ok"
    error_type: str | None = None
    error_detail: str | None = None
    session: Session | None = None
    first_executed_at: float | None = None

    def _capture_usage(event: object) -> None:
        event_usage = getattr(event, "usage", None)
        if event_usage is None:
            return
        usage["prompt_tokens"] = event_usage.prompt_tokens
        usage["completion_tokens"] = event_usage.completion_tokens
        usage["total_tokens"] = event_usage.total_tokens
        details = getattr(event_usage, "completion_tokens_details", None)
        usage["reasoning_tokens"] = (
            getattr(details, "reasoning_tokens", None) if details is not None else None
        )

    try:
        async with asyncio.timeout(CASE_TIMEOUT_SECONDS):
            stream = await client.chat.completions.create(
                model=config.model,
                stream=True,
                seed=config.seed,
                temperature=config.temperature,
                # `stream_options` must be omitted entirely rather than
                # passed as `None`: the openai SDK serializes an explicit
                # `None` as `"stream_options": null`, which OpenRouter's
                # schema validator rejects with a 400 (it wants either an
                # object or the key absent).
                stream_options=stream_options if stream_options is not None else omit,  # type: ignore
                messages=messages,  # type: ignore
            )
            try:
                if strategy is Strategy.INCREMENTAL:
                    session = create_python_session()
                    async with session:
                        driver = StreamDriver(session, markdown_code("python"))
                        async for event in stream:
                            _capture_usage(event)
                            if not event.choices:
                                continue
                            delta = event.choices[0].delta.content
                            if delta is None:
                                continue
                            completion_chunks += 1
                            for chunk in await driver.apush(delta):
                                output += chunk
                            if driver.has_error:
                                early_exit = True
                                break
                        for chunk in await driver.afinish():
                            output += chunk
                else:
                    raw = ""
                    async for event in stream:
                        _capture_usage(event)
                        if not event.choices:
                            continue
                        delta = event.choices[0].delta.content
                        if delta is None:
                            continue
                        completion_chunks += 1
                        raw += delta

                    code = _extract_code_block(raw)
                    # A dedicated executor per case, matching the fresh
                    # session (and therefore fresh executor) the incremental
                    # branch also gets per case -- neither strategy carries
                    # REPL state between rows. `timeout` matches
                    # `CASE_TIMEOUT_SECONDS` rather than relying on the
                    # coincidence that it's also `InProcPythonExecutor`'s
                    # own default.
                    async with InProcPythonExecutor(
                        timeout=CASE_TIMEOUT_SECONDS
                    ) as executor:
                        result = await executor.aexecute(code)
                    # Stamped unconditionally -- including when execution
                    # failed -- so sequential's "first statement" stays
                    # symmetric with jitgen's own Session, which records
                    # first_statement_at in a finally block for the same
                    # reason: excluding a failing run would flatter exactly
                    # the runs that failed fastest.
                    first_executed_at = time.perf_counter()
                    if not result.success:
                        raise ExecutionError.from_result(result)
                    output = result.output or ""
            finally:
                await stream.close()
    except TimeoutError:
        status = "timeout"
        error_type = "TimeoutError"
        error_detail = f"case exceeded {CASE_TIMEOUT_SECONDS}s"
    except JitGenError as exc:
        status = "timeout" if exc.has_timed_out else "error"
        error_detail = str(exc)
        error_type = _error_type(error_detail, type(exc).__name__)
        # `raise ExecutionError.from_result(result)` above happens before
        # `output` is assigned on the failure path, so without this,
        # sequential would silently drop any stdout a failing block
        # produced before it raised -- while incremental keeps it, since it
        # accumulates `output` per statement as it goes. `exc.output` is set
        # by `ExecutionError.from_result`/jitgen's own execution path either way.
        if not output:
            output = exc.output
    except ValueError as exc:
        # The only `ValueError` reachable here is `_extract_code_block`'s own
        # "no code block found" -- a harness-level failure to extract
        # anything to run, not a runtime error from executed model code (a
        # `ValueError` raised *by* model code would already have been caught
        # above, wrapped in `ExecutionError`/`JitGenError`, by
        # `InProcPythonExecutor`). Labeled distinctly from a genuine
        # `ValueError` so `build_artifacts.py` can classify it as a
        # generation-level failure instead of a model-code exception type.
        status = "error"
        error_detail = str(exc)
        error_type = "GenerationError"
    except Exception as exc:  # noqa: BLE001 - any other failure must be classified, not raised
        status = "error"
        error_detail = str(exc)
        error_type = type(exc).__name__

    if strategy is Strategy.INCREMENTAL:
        first_statement_seconds = (
            session.stats.since(started) if session is not None else None
        )
    else:
        first_statement_seconds = (
            first_executed_at - started if first_executed_at is not None else None
        )

    actual_output = _actual_output(output) if status == "ok" else ""
    correct_output = (
        case.example.is_correct_answer(actual_output) if status == "ok" else False
    )

    return RunResult(
        dataset_row=case.dataset_row,
        case_index=case.case_index,
        task_id=case.example.task_id,
        strategy=strategy,
        config=config,
        status=status,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - started,
        first_statement_seconds=first_statement_seconds,
        completion_chunks=completion_chunks,
        api_prompt_tokens=usage["prompt_tokens"],
        api_completion_tokens=usage["completion_tokens"],
        api_reasoning_tokens=usage["reasoning_tokens"],
        api_total_tokens=usage["total_tokens"],
        early_exit=early_exit,
        output=output,
        actual_output=actual_output,
        expected_output=case.example.test_output,
        correct_output=correct_output,
        error_type=error_type,
        error_detail=error_detail,
    )


async def warm_up(
    client: AsyncOpenAI,
    config: RunConfig,
    case: IndexedCase,
    messages: list[dict[str, str]],
    strategy: Strategy,
    stream_options: dict[str, bool] | None,
) -> None:
    """Absorb cold-start/first-token cost before any case in this pass is timed.

    Runs the dataset's first row through the exact same `run_case` path as
    every timed case (streaming, grammar-aware extraction for incremental,
    a real execution) and discards the result, rather than a synthetic
    prompt that would leave e.g. the Lark parser cold. Excluded from
    measurement entirely, and called identically at the start of every pass
    (for providers with local residency) so cold-start variance never gets
    charged to whichever strategy happens to run first.

    Args:
        client: The chat completions client to warm up.
        config: The run configuration.
        case: The dataset row to warm up with -- the run's first row.
        messages: `case` rendered to chat messages.
        strategy: Which execution strategy this pass measures.
        stream_options: Provider-specific `stream_options` for the request.
    """
    await run_case(client, config, case, messages, strategy, stream_options)
