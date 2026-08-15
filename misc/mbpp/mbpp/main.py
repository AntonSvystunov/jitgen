import asyncio
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import httpx
from dotenv import load_dotenv
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

from .dataset import IndexedCase, load_indexed_dataset
from .models import MODELS, ModelSpec
from .prompts import PROMPT_VERSION, render
from .providers import ProviderUnavailableError
from .results import (
    Strategy,
    assert_unique_csv_stems,
    csv_path_for,
    to_csv_row,
    write_csv,
)

# Fixed so decoding is deterministic: same model, same seed, same
# temperature, same prompt should yield the same completion, so any
# difference between strategies is attributable to execution strategy, not
# sampling noise.
SEED = 42
TEMPERATURE = 0.0
ROWS_TO_RUN = 113
CASE_TIMEOUT_SECONDS = 60.0
WARMUP_MESSAGES = [{"role": "user", "content": "hi"}]

# A literal, not a reference to `PROMPT_VERSION` — pinning it to the same
# name it's checked against would make the check vacuous. Computed from the
# templates in `prompts.py` as of writing; if it no longer matches, either a
# prompt edit is unreviewed or this literal needs updating to acknowledge it.
EXPECTED_PROMPT_VERSION = "6601f9c33052"

# Lazy capture so a closing fence, when present, stops the match at the
# first one rather than swallowing trailing prose. `\Z` as the alternative
# closer means a response cut off (or that never closes the fence) still
# matches, capturing everything to the end of the response instead of
# failing to match at all.
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)(?:\n?```|\Z)", re.DOTALL)

# Pulls the leading class name off jitgen's own `"ClassName: message"`
# error formatting (`jitgen.executors.python._describe`,
# `jitgen.session._describe`).
_ERROR_TYPE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")

Status = Literal["ok", "timeout", "error"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The parameters that must be identical across both strategies' passes."""

    model: str
    seed: int
    temperature: float
    prompt_version: str


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


async def _warm_up(client: AsyncOpenAI, config: RunConfig) -> None:
    """Absorb cold-start/first-token cost before any case in this pass is timed.

    Excluded from measurement entirely, and called identically at the start
    of every pass (for providers with local residency) so cold-start
    variance never gets charged to whichever strategy happens to run first.

    Args:
        client: The chat completions client to warm up.
        config: The run configuration (only `model`/`seed`/`temperature` are used).
    """
    await client.chat.completions.create(
        model=config.model,
        stream=False,
        seed=config.seed,
        temperature=config.temperature,
        messages=WARMUP_MESSAGES,  # type: ignore
    )


def _extract_code_block(raw: str) -> str:
    """Pull the fenced ```python code block out of a full model response.

    Deliberately independent of jitgen's `MarkerSegmenter`: the sequential
    baseline is meant to be a plain generate-then-execute pipeline with none
    of jitgen's own extraction machinery in the loop, so it gets its own
    minimal regex-based extractor instead of reusing `markdown_code`.

    A missing closing fence is not an error — the model may have been cut
    off mid-block, or omitted it entirely — everything from the opening
    fence to the end of the response is used instead. A missing *opening*
    fence is still an error, since then there is no code to run at all.

    Args:
        raw: The full, already-collected model response.

    Returns:
        The code inside the fence.

    Raises:
        ValueError: `raw` contains no ` ```python ` opening fence.
    """
    match = _CODE_BLOCK_RE.search(raw)
    if match is None:
        msg = "no ```python code block found in response"
        raise ValueError(msg)
    return match.group(1)


async def _run_case(
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
        status = "error"
        error_detail = str(exc)
        error_type = "ValueError"
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


def _print_result(case: IndexedCase, result: RunResult) -> None:
    marker = {"ok": "ok", "timeout": "TIMEOUT", "error": "ERROR"}[result.status]
    print(
        f"[{result.strategy}] row={result.dataset_row} task_id={case.example.task_id} "
        f"{marker} {result.elapsed_seconds:.3f}s tokens={result.completion_chunks} "
        f"-> {result.output!r}"
    )
    if result.error_detail:
        print(f"    {result.error_detail}")


def _by_row(results: Iterable[RunResult]) -> dict[int, dict[Strategy, RunResult]]:
    """Group results by dataset row, then by strategy, for cross-strategy pairing."""
    grouped: dict[int, dict[Strategy, RunResult]] = defaultdict(dict)
    for result in results:
        grouped[result.dataset_row][result.strategy] = result
    return grouped


def _print_totals(results: list[RunResult]) -> None:
    """Print aggregate timing over all rows and over rows where both strategies succeeded."""
    rows = list(_by_row(results).values())

    def totals(subset: Iterable[dict[Strategy, RunResult]]) -> tuple[float, float]:
        incremental = sum(r[Strategy.INCREMENTAL].elapsed_seconds for r in subset)
        sequential = sum(r[Strategy.SEQUENTIAL].elapsed_seconds for r in subset)
        return incremental, sequential

    incremental_total, sequential_total = totals(rows)
    print(f"overall total incremental: {incremental_total:.3f}s")
    print(f"overall total sequential:  {sequential_total:.3f}s")
    if incremental_total:
        print(f"overall speedup: {sequential_total / incremental_total:.2f}x")

    # An aborted pass finishes fast because it aborted, so the headline
    # speedup above is confounded by any row where one strategy failed and
    # the other didn't. This subset isolates the comparison that's actually
    # apples-to-apples.
    matched = [
        r
        for r in rows
        if r.get(Strategy.INCREMENTAL) is not None
        and r.get(Strategy.SEQUENTIAL) is not None
        and r[Strategy.INCREMENTAL].status == "ok"
        and r[Strategy.SEQUENTIAL].status == "ok"
    ]
    print(f"rows where both strategies succeeded: {len(matched)}/{len(rows)}")
    if matched:
        incremental_matched, sequential_matched = totals(matched)
        print(f"matched total incremental: {incremental_matched:.3f}s")
        print(f"matched total sequential:  {sequential_matched:.3f}s")
        if incremental_matched:
            print(f"matched speedup: {sequential_matched / incremental_matched:.2f}x")


def _print_h1(results: list[RunResult]) -> None:
    """H1: on rows both strategies solved cleanly, does incremental's first output land earlier?

    A paired per-row delta rather than summed totals -- at a small sample
    size, one slow row would otherwise dominate a sum and misrepresent the
    typical case.
    """
    deltas: list[float] = []
    for pair in _by_row(results).values():
        inc = pair.get(Strategy.INCREMENTAL)
        seq = pair.get(Strategy.SEQUENTIAL)
        if inc is None or seq is None or inc.status != "ok" or seq.status != "ok":
            continue
        if inc.first_statement_seconds is None or seq.first_statement_seconds is None:
            continue
        deltas.append(seq.first_statement_seconds - inc.first_statement_seconds)

    print(f"H1 (first-output latency): {len(deltas)} rows compared")
    if not deltas:
        return
    deltas.sort()
    mean = sum(deltas) / len(deltas)
    median = deltas[len(deltas) // 2]
    wins = sum(1 for delta in deltas if delta > 0)
    print(f"  mean advantage: {mean:+.3f}s, median: {median:+.3f}s")
    print(f"  incremental first on {wins}/{len(deltas)} rows")


def _print_h2(results: list[RunResult]) -> None:
    """H2: on rows incremental errored, does its early-exit save tokens/time over sequential?

    Split by `early_exit` -- a non-early-exit error row isn't the mechanism
    H2 claims drives the saving (the model's code was well-formed enough to
    run to its last statement before failing), so averaging it in with
    genuine early-exits would dilute the effect being measured.
    """
    buckets: dict[bool, list[tuple[RunResult, RunResult]]] = defaultdict(list)
    for pair in _by_row(results).values():
        inc = pair.get(Strategy.INCREMENTAL)
        seq = pair.get(Strategy.SEQUENTIAL)
        if inc is None or seq is None or inc.status != "error":
            continue
        buckets[inc.early_exit].append((inc, seq))

    print("H2 (failure-path latency):")
    if not buckets:
        print("  no rows where incremental errored")
        return
    for early_exit, pairs in sorted(buckets.items(), key=lambda item: not item[0]):
        label = "early-exit" if early_exit else "no early-exit"
        inc_time = sum(inc.elapsed_seconds for inc, _ in pairs)
        seq_time = sum(seq.elapsed_seconds for _, seq in pairs)
        inc_tokens = sum(inc.completion_chunks for inc, _ in pairs)
        seq_tokens = sum(seq.completion_chunks for _, seq in pairs)
        print(
            f"  {label}: {len(pairs)} rows -- "
            f"time inc={inc_time:.3f}s seq={seq_time:.3f}s, "
            f"tokens inc={inc_tokens} seq={seq_tokens}"
        )


def _print_summary(results: list[RunResult]) -> None:
    """Print per-model totals plus the H1/H2 hypothesis comparisons.

    Args:
        results: Every `RunResult` from every model/strategy pass.
    """
    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        by_model[result.config.model].append(result)

    for model, model_results in by_model.items():
        print(f"=== summary: {model} ===")
        _print_totals(model_results)
        print()
        _print_h1(model_results)
        print()
        _print_h2(model_results)
        print()


async def _run_pass(
    spec: ModelSpec,
    client: AsyncOpenAI,
    http: httpx.AsyncClient,
    config: RunConfig,
    strategy: Strategy,
    rendered: list[tuple[IndexedCase, list[dict[str, str]]]],
) -> list[RunResult]:
    """Run every case for one (model, strategy) pass, bracketed by prepare/release.

    Args:
        spec: The model under test.
        client: The chat completions client to run cases through.
        http: Client scoped to `spec.provider.native_base_url`.
        config: The pinned run configuration for this model.
        strategy: Which execution strategy this pass measures.
        rendered: Every case paired with its rendered messages.

    Returns:
        One `RunResult` per case, in row order.
    """
    print(f"--- pass: {strategy} ---")
    await spec.provider.prepare(http, spec.model)
    pass_results: list[RunResult] = []
    try:
        if spec.provider.needs_warmup:
            await _warm_up(client, config)
        for case, messages in rendered:
            result = await _run_case(
                client, config, case, messages, strategy, spec.provider.stream_options()
            )
            pass_results.append(result)
            _print_result(case, result)
    finally:
        await spec.provider.release(http, spec.model)
    return pass_results


async def run_evaluation() -> None:
    """Compare incremental vs. sequential execution over every configured model.

    Each model's strategy runs as one full pass over every row before the
    other strategy starts — never interleaved — with the model unloaded and
    reloaded fresh at the start of each pass and released at the end, so
    neither pass inherits KV-cache state or load effects from the other.
    Each (model, strategy) pass writes its own CSV under `misc/mbpp/results/`.
    """
    if PROMPT_VERSION != EXPECTED_PROMPT_VERSION:
        msg = (
            f"prompt template changed: expected version {EXPECTED_PROMPT_VERSION!r}, "
            f"got {PROMPT_VERSION!r}. Update EXPECTED_PROMPT_VERSION if intentional."
        )
        raise RuntimeError(msg)

    assert_unique_csv_stems(MODELS)

    indexed_cases = load_indexed_dataset()[:ROWS_TO_RUN]
    # Rendered once and reused verbatim by both passes: there is no
    # per-strategy templating path that could drift.
    rendered = [(case, render(case.example)) for case in indexed_cases]

    all_results: list[RunResult] = []
    for spec in MODELS:
        try:
            client = spec.provider.build_client()
        except ProviderUnavailableError as exc:
            print(f"skipping {spec.label}: {exc}\n")
            continue

        async with spec.provider.build_native_client() as http:
            await spec.provider.verify_model_present(http, spec.model)
            config = RunConfig(
                model=spec.model,
                seed=SEED,
                temperature=TEMPERATURE,
                prompt_version=PROMPT_VERSION,
            )
            print(f"=== model: {spec.label} ({spec.provider.name}) ===")

            for strategy in (Strategy.INCREMENTAL, Strategy.SEQUENTIAL):
                pass_results = await _run_pass(
                    spec, client, http, config, strategy, rendered
                )
                write_csv(
                    csv_path_for(spec, strategy),
                    [to_csv_row(result, spec) for result in pass_results],
                )
                all_results.extend(pass_results)
                print()

    if all_results:
        _print_summary(all_results)


def main() -> None:
    """Entry point for the MBPP evaluation script."""
    load_dotenv()
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
