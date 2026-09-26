# Benchmarks jitgen's incremental PTC dispatch against "regular" PTC (buffer the
# whole tool call, then execute it), for both the Python and JavaScript
# executors. One real tool call is recorded per language, then both strategies
# replay its exact delta timing, so only dispatch timing differs.
import argparse
import asyncio
import json
import os
import statistics
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, aclosing
from dataclasses import dataclass

import js_example
import python_example
from dotenv import load_dotenv
from jitgen import ExecutorBase, JitGenError, Session, StreamDriver
from jitgen_openai import OpenAIToolCallSegmenter
from utils import build_client, create_eval_stream, iter_argument_deltas, require_model

JITGEN_LABEL = "jitgen (incremental PTC)"
BASELINE_LABEL = "regular PTC (buffer, then execute)"


@dataclass(frozen=True)
class _Language:
    """Everything needed to benchmark one guest language.

    Attributes:
        name: Display name.
        system_prompt: The system message describing the `eval` tool.
        user_task: The task sent to the model.
        tool: The `eval` tool definition.
        make_executor: Builds a fresh executor for the baseline strategy.
        open_session: Opens a fresh session for the jitgen strategy.
    """

    name: str
    system_prompt: str
    user_task: str
    tool: dict[str, object]
    make_executor: Callable[[], ExecutorBase]
    open_session: Callable[[], AbstractAsyncContextManager[Session]]


LANGUAGES = {
    "python": _Language(
        name="Python",
        system_prompt=python_example.SYSTEM_PROMPT,
        user_task=python_example.USER_TASK,
        tool=python_example.EVAL_TOOL,
        make_executor=python_example.make_executor,
        open_session=python_example.open_session,
    ),
    "js": _Language(
        name="JavaScript",
        system_prompt=js_example.SYSTEM_PROMPT,
        user_task=js_example.USER_TASK,
        tool=js_example.EVAL_TOOL,
        make_executor=js_example.make_executor,
        open_session=js_example.open_session,
    ),
}


@dataclass
class _Delta:
    """One recorded `arguments` delta.

    Attributes:
        delay: Seconds since the previous delta (or the stream start).
        text: The delta's raw text.
    """

    delay: float
    text: str


@dataclass
class _Trial:
    """One replay's measured timings.

    Attributes:
        time_to_first_output: Seconds until the first execution output, or
            `None` if there was none.
        total_time: Seconds until all code finished executing.
        output: The concatenated captured output.
    """

    time_to_first_output: float | None
    total_time: float
    output: str


@dataclass
class _LanguageResult:
    """One language's recorded stream and trial results.

    Attributes:
        language: The benchmarked language.
        generation_seconds: Total recorded generation time.
        delta_count: Number of recorded non-empty `arguments` deltas.
        jitgen_trials: Results of the jitgen strategy.
        baseline_trials: Results of the baseline strategy.
    """

    language: _Language
    generation_seconds: float
    delta_count: int
    jitgen_trials: list[_Trial]
    baseline_trials: list[_Trial]


async def _record_stream(model: str, language: _Language) -> list[_Delta]:
    """Make one real streamed `eval` tool call and record its delta timing.

    Args:
        model: The model identifier to request.
        language: The language whose prompt and tool to use.

    Returns:
        The `arguments` deltas in arrival order.

    Raises:
        RuntimeError: If the model never produced a tool call.
    """
    messages: list[dict[str, object]] = [
        {"role": "system", "content": language.system_prompt},
        {"role": "user", "content": language.user_task},
    ]
    deltas: list[_Delta] = []
    saw_tool_call = False

    stream = await create_eval_stream(build_client(), model, messages, language.tool)
    last = time.monotonic()
    async with stream, aclosing(iter_argument_deltas(stream)) as argument_deltas:
        async for call_id, text in argument_deltas:
            saw_tool_call = saw_tool_call or call_id is not None
            if not text:
                continue
            now = time.monotonic()
            deltas.append(_Delta(delay=now - last, text=text))
            last = now

    if not saw_tool_call or not deltas:
        msg = f"model did not call the eval tool ({language.name})"
        raise RuntimeError(msg)
    return deltas


async def _replay_jitgen(language: _Language, deltas: list[_Delta]) -> _Trial:
    """Replay `deltas` through jitgen, executing statements as they complete.

    Args:
        language: The language whose session executes the code.
        deltas: The recorded `arguments` deltas to replay.

    Returns:
        The trial's measured timings.
    """
    captured: list[str] = []
    time_to_first_output: float | None = None
    start = time.monotonic()

    def capture(outputs: list[str]) -> None:
        nonlocal time_to_first_output
        if time_to_first_output is None and any(outputs):
            time_to_first_output = time.monotonic() - start
        captured.extend(outputs)

    async with language.open_session() as session:
        driver = StreamDriver(session, OpenAIToolCallSegmenter(property_name="code"))
        for delta in deltas:
            await asyncio.sleep(delta.delay)
            capture(await driver.apush(delta.text))
        try:
            capture(await driver.afinish())
        except JitGenError as exc:
            captured.append(f"execution error: {exc}")

    return _Trial(time_to_first_output, time.monotonic() - start, "".join(captured))


async def _replay_baseline(language: _Language, deltas: list[_Delta]) -> _Trial:
    """Replay `deltas` the regular way: buffer them all, then execute once.

    Uses the same executor as the jitgen replay, so only dispatch timing
    differs.

    Args:
        language: The language whose executor runs the code.
        deltas: The recorded `arguments` deltas to replay.

    Returns:
        The trial's measured timings.
    """
    start = time.monotonic()
    async with language.make_executor() as executor:
        raw_parts: list[str] = []
        for delta in deltas:
            await asyncio.sleep(delta.delay)
            raw_parts.append(delta.text)
        code = json.loads("".join(raw_parts))["code"]
        result = await executor.aexecute(code)

    total_time = time.monotonic() - start
    output = result.output or ""
    if result.error:
        output += f"execution error: {result.error}"
    return _Trial(total_time if result.output else None, total_time, output)


def _fmt(seconds: float | None) -> str:
    """Render `seconds` as `"1.23s"`, or `"n/a"` when `None`."""
    return f"{seconds:.2f}s" if seconds is not None else "n/a"


def _fmt_mean(values: list[float]) -> str:
    """Render the mean of `values`, with the stdev when there are several."""
    spread = f" (± {statistics.stdev(values):.2f}s)" if len(values) > 1 else ""
    return f"mean {statistics.mean(values):.2f}s{spread}"


async def _run_trials(
    label: str,
    replay: Callable[[], Awaitable[_Trial]],
    trials: int,
) -> list[_Trial]:
    """Run `replay` `trials` times, printing each trial's timings.

    Args:
        label: Strategy name, printed as a header.
        replay: Runs one replay of the strategy being benchmarked.
        trials: Number of replays.

    Returns:
        Every trial's result, in order.
    """
    print(f"\n=== {label} ===")
    results: list[_Trial] = []
    for i in range(trials):
        trial = await replay()
        results.append(trial)
        print(
            f"  trial {i + 1}/{trials}: "
            f"time-to-first-output={_fmt(trial.time_to_first_output)}  "
            f"total={_fmt(trial.total_time)}"
        )
    return results


async def _benchmark_language(
    model: str, language: _Language, trials: int
) -> _LanguageResult:
    """Record one tool call for `language`, then replay it with both strategies.

    Args:
        model: The model identifier to request.
        language: The language to benchmark.
        trials: Number of replays per strategy.

    Returns:
        The recorded stream's stats and every trial's result.
    """
    print(f"\n##### {language.name} #####")
    print("Recording one real streamed `eval` tool call...")
    deltas = await _record_stream(model, language)
    generation_seconds = sum(d.delay for d in deltas)
    print(
        f"Recorded {len(deltas)} argument deltas over "
        f"{generation_seconds:.2f}s of generation."
    )

    jitgen_trials = await _run_trials(
        JITGEN_LABEL, lambda: _replay_jitgen(language, deltas), trials
    )
    baseline_trials = await _run_trials(
        BASELINE_LABEL, lambda: _replay_baseline(language, deltas), trials
    )
    return _LanguageResult(
        language, generation_seconds, len(deltas), jitgen_trials, baseline_trials
    )


def _summarize_strategy(label: str, trials: list[_Trial]) -> None:
    """Print a strategy's mean timings.

    Args:
        label: Strategy name.
        trials: That strategy's trial results.
    """
    ttfos = [
        t.time_to_first_output for t in trials if t.time_to_first_output is not None
    ]
    print(f"\n  {label}:")
    print(f"    time to first output: {_fmt_mean(ttfos) if ttfos else 'n/a'}")
    print(f"    total time:           {_fmt_mean([t.total_time for t in trials])}")


def _summarize_language(result: _LanguageResult) -> None:
    """Print one language's timings, speedup, and output consistency.

    Args:
        result: The language's benchmark result.
    """
    print(
        f"\n{result.language.name}: {result.delta_count} deltas over "
        f"{result.generation_seconds:.2f}s of recorded generation"
    )
    _summarize_strategy(JITGEN_LABEL, result.jitgen_trials)
    _summarize_strategy(BASELINE_LABEL, result.baseline_trials)

    jitgen_total = statistics.mean(t.total_time for t in result.jitgen_trials)
    baseline_total = statistics.mean(t.total_time for t in result.baseline_trials)
    print(
        f"\n  Speedup (total time): {baseline_total / jitgen_total:.2f}x "
        f"({baseline_total - jitgen_total:.2f}s saved on average)"
    )
    outputs = {t.output for t in result.jitgen_trials + result.baseline_trials}
    print(f"  Outputs identical across every trial and strategy: {len(outputs) == 1}")


def _read_trial_count() -> int:
    """Read the replay count from `BENCHMARK_TRIALS`.

    Returns:
        The number of replays per strategy.

    Raises:
        ValueError: If `BENCHMARK_TRIALS` is not a positive integer.
    """
    trials = int(os.environ.get("BENCHMARK_TRIALS", "5"))
    if trials < 1:
        msg = f"BENCHMARK_TRIALS must be at least 1, got {trials}"
        raise ValueError(msg)
    return trials


def _parse_languages() -> list[_Language]:
    """Parse which languages to benchmark from the command line.

    Returns:
        The selected languages, in the order given (all of them by default).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "languages",
        nargs="*",
        choices=list(LANGUAGES),
        help="languages to benchmark (default: all)",
    )
    names = parser.parse_args().languages or list(LANGUAGES)
    return [LANGUAGES[name] for name in dict.fromkeys(names)]


async def main() -> None:
    languages = _parse_languages()
    load_dotenv()
    model = require_model()
    trials = _read_trial_count()

    results = [
        await _benchmark_language(model, language, trials) for language in languages
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results:
        _summarize_language(result)


if __name__ == "__main__":
    asyncio.run(main())
