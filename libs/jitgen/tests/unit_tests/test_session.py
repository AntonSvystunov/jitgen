import asyncio

import pytest

from jitgen import ExecutionError, ExtractionError, Session, SessionStats
from jitgen.session import _describe, _may_close_statement

from .conftest import FakeExecutor, FakeExtractor

TIMEOUT = 1.0


def test_since_returns_elapsed_time_when_a_statement_ran():
    stats = SessionStats(first_statement_at=15.0)

    assert stats.since(10.0) == 5.0


def test_since_returns_default_when_nothing_ran():
    stats = SessionStats()

    assert stats.since(10.0) is None
    assert stats.since(10.0, default=-1.0) == -1.0


def test_reset_zeroes_every_field():
    stats = SessionStats(
        statements_executed=3,
        statements_failed=1,
        first_statement_at=1.0,
        last_statement_at=2.0,
        execution_seconds=0.5,
        early_exit=True,
    )

    stats.reset()

    assert stats == SessionStats()


def test_describe_includes_type_and_message():
    assert _describe(ValueError("boom")) == "ValueError: boom"


def test_describe_omits_colon_when_message_is_empty():
    assert _describe(ValueError()) == "ValueError"


@pytest.mark.parametrize(
    ("buffer", "appended_at", "expected"),
    [
        pytest.param("hello\n", 0, True, id="newline-in-appended-span"),
        pytest.param("x", 0, True, id="non-blank-at-buffer-start"),
        pytest.param("a\nb", 2, True, id="non-blank-right-after-newline"),
        pytest.param("   \n", 0, True, id="newline-after-leading-whitespace"),
        pytest.param("  x", 0, False, id="non-blank-mid-line-no-newline"),
        pytest.param("ab", 1, False, id="appended-char-mid-line-no-newline"),
    ],
)
def test_may_close_statement(buffer: str, appended_at: int, expected: bool):
    assert _may_close_statement(buffer, appended_at) is expected


async def test_executor_property_returns_the_configured_executor(
    session: Session, executor: FakeExecutor
):
    assert session.executor is executor


async def test_push_with_no_boundary_inducing_text_does_not_trigger_extraction(
    session: Session, executor: FakeExecutor
):
    # Mid-line text with no newline and not at column 0 cannot possibly close
    # a statement, so push() must skip extraction entirely as a latency guard.
    session.push("a")
    session.push("b")

    assert session.buffer == "ab"
    assert executor.calls == []


async def test_a_later_queued_statement_is_skipped_after_an_earlier_one_fails(
    session: Session, executor: FakeExecutor
):
    executor.raise_for("a", ValueError("boom"))
    executor.respond("b", output="B")

    session.push("a\nb\n")  # both statements are queued by the same extraction

    with pytest.raises(ExecutionError):
        await session.result()

    # P8 (fail fast): "b" was already queued alongside "a", but the worker
    # must never execute it once the failure from "a" is recorded.
    assert executor.calls == ["a"]


async def test_reset_tolerates_an_executor_that_cannot_be_cancelled(
    session: Session, executor: FakeExecutor
):
    executor.raise_on_acancel = RuntimeError("this executor cannot cancel")
    session.push("a\n")

    await asyncio.wait_for(session.reset(), TIMEOUT)  # must not raise or hang

    assert session.has_error is False


async def test_cancelling_result_while_awaiting_execution_requests_cancellation(
    session: Session, executor: FakeExecutor
):
    executor.gate("a")  # deliberately never released
    started = executor.started("a")
    session.push("a\n")
    await asyncio.wait_for(started.wait(), TIMEOUT)

    result_task = asyncio.ensure_future(session.result())
    await asyncio.sleep(0)  # let it reach the quiescence wait
    result_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await result_task

    # Cancellation is expected to trigger a best-effort request to interrupt
    # the in-flight statement, even though the cancelled task can't await it.
    for _ in range(100):
        if executor.acancel_calls:
            break
        await asyncio.sleep(0)
    assert executor.acancel_calls == 1


async def test_push_dispatches_statements_and_result_returns_output_in_order(
    session: Session, executor: FakeExecutor
):
    executor.respond("a", output="A")
    executor.respond("b", output="B")

    session.push("a\nb\n")

    assert await session.result() == "AB"
    assert executor.calls == ["a", "b"]
    assert session.stats.statements_executed == 2
    assert session.stats.statements_failed == 0


async def test_push_after_result_invalidates_the_cached_output(
    session: Session, executor: FakeExecutor
):
    executor.respond("a", output="A")
    executor.respond("b", output="B")

    session.push("a\n")
    assert await session.result() == "A"

    session.push("b\n")
    assert await session.result() == "B"


async def test_result_is_idempotent_on_success(
    session: Session, executor: FakeExecutor
):
    executor.respond("a", output="A")
    session.push("a\n")

    first = await session.result()
    second = await session.result()

    assert first == second == "A"
    assert executor.calls == ["a"]  # not re-dispatched by the second call


async def test_result_is_idempotent_on_failure(
    session: Session, executor: FakeExecutor
):
    executor.raise_for("a", ValueError("boom"))
    session.push("a\n")

    with pytest.raises(ExecutionError) as first:
        await session.result()
    with pytest.raises(ExecutionError) as second:
        await session.result()

    assert first.value is second.value


async def test_take_output_returns_only_output_produced_since_the_last_call(
    session: Session, executor: FakeExecutor
):
    executor.respond("a", output="A")
    gate_b = executor.gate("b")
    started_b = executor.started("b")

    session.push("a\nb\n")
    await asyncio.wait_for(started_b.wait(), TIMEOUT)

    # "a" already ran to completion (its output is available) while "b" sits
    # blocked on its gate -- push() never had to wait for either of them.
    assert executor.calls == ["a", "b"]
    assert session.take_output() == "A"
    assert session.take_output() == ""

    gate_b.set()
    assert await session.result() == ""  # "b" produced no output of its own


async def test_extraction_error_is_recorded_synchronously_by_push(
    session: Session, extractor: FakeExtractor, executor: FakeExecutor
):
    session.push(extractor.SYNTAX_ERROR_LINE + "\n")

    assert session.has_error
    assert isinstance(session.error, ExtractionError)
    assert executor.calls == []  # nothing was ever dispatched


async def test_push_is_a_noop_once_an_error_is_recorded(
    session: Session, extractor: FakeExtractor, executor: FakeExecutor
):
    session.push(extractor.SYNTAX_ERROR_LINE + "\n")
    buffer_after_error = session.buffer

    session.push("a\n")

    assert session.buffer == buffer_after_error
    assert executor.calls == []


async def test_extraction_error_raises_from_result(
    session: Session, extractor: FakeExtractor
):
    session.push(extractor.SYNTAX_ERROR_LINE + "\n")

    with pytest.raises(ExtractionError):
        await session.result()


async def test_execution_exception_is_wrapped_and_raised_from_result(
    session: Session, executor: FakeExecutor
):
    original = ValueError("boom")
    executor.raise_for("a", original)
    session.push("a\n")

    with pytest.raises(ExecutionError) as excinfo:
        await session.result()

    assert excinfo.value.statement == "a"
    assert excinfo.value.__cause__ is original
    assert "boom" in str(excinfo.value)
    assert session.stats.statements_failed == 1


async def test_failed_execution_result_is_wrapped_and_output_is_not_kept(
    session: Session, executor: FakeExecutor
):
    executor.respond("a", success=False, error="bad thing", output="partial")
    session.push("a\n")

    with pytest.raises(ExecutionError) as excinfo:
        await session.result()

    assert str(excinfo.value) == "bad thing"
    assert excinfo.value.output == "partial"
    assert session.take_output() == ""  # the partial output is not surfaced


async def test_reset_clears_state_but_preserves_stats(
    session: Session, executor: FakeExecutor
):
    executor.respond("a", output="A")
    session.push("a\n")
    assert await session.result() == "A"

    # Queued but never awaited: the worker cannot have picked "b" up before
    # reset() synchronously drains the queue below.
    session.push("b\n")
    await session.reset()

    assert session.buffer == ""
    assert session.has_error is False
    assert executor.calls == ["a"]  # "b" was drained before it ever ran
    assert executor.acancel_calls == 1
    assert session.stats.statements_executed == 1  # preserved, not cleared


async def test_reset_discards_a_stale_in_flight_failure(
    session: Session, executor: FakeExecutor
):
    gate = executor.gate("a")
    started = executor.started("a")
    session.push("a\n")
    await asyncio.wait_for(started.wait(), TIMEOUT)

    # acancel() (called by reset()) releases the gate, so the in-flight call
    # only fails *after* reset() has already bumped the generation counter.
    executor.raise_for("a", ValueError("late failure"))
    await asyncio.wait_for(session.reset(), TIMEOUT)

    assert session.has_error is False
    assert session.buffer == ""
    assert await session.result() == ""
    assert gate.is_set()  # sanity: the stale failure really did fire


async def test_session_remains_usable_after_aclose(
    session: Session, executor: FakeExecutor
):
    executor.respond("a", output="A")
    session.push("a\n")
    assert await session.result() == "A"

    await session.aclose()

    # A later push must start a fresh worker rather than leave the session
    # permanently unusable.
    executor.respond("b", output="B")
    session.push("b\n")
    assert await session.result() == "B"


async def test_aclose_closes_an_owned_executor(
    extractor: FakeExtractor, executor: FakeExecutor
):
    session = Session(extractor=extractor, executor=executor, owns_executor=True)

    await session.aclose()

    assert executor.aclose_calls == 1


async def test_aclose_does_not_close_a_foreign_executor(
    extractor: FakeExtractor, executor: FakeExecutor
):
    session = Session(extractor=extractor, executor=executor, owns_executor=False)

    await session.aclose()

    assert executor.aclose_calls == 0


async def test_aclose_cancels_in_flight_work_without_waiting_for_it(
    session: Session, executor: FakeExecutor
):
    executor.gate("a")  # deliberately never released
    started = executor.started("a")
    session.push("a\n")
    await asyncio.wait_for(started.wait(), TIMEOUT)

    await asyncio.wait_for(session.aclose(), TIMEOUT)

    assert executor.acancel_calls == 0  # aclose() only cancels the asyncio task


async def test_context_manager_closes_the_session_on_exit(
    extractor: FakeExtractor, executor: FakeExecutor
):
    async with Session(extractor=extractor, executor=executor, owns_executor=True):
        pass

    assert executor.aclose_calls == 1
