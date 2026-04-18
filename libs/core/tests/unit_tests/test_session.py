"""Tests for jitgen_core.session.Session."""

import asyncio

import pytest

from jitgen_core import ExecutionResult, Session
from jitgen_core.base import BaseExecutor, SourceCode, StatementExtractor


# ── Fakes ──────────────────────────────────────────────────────────────────────

class EchoExtractor:
    """Trivial extractor: each newline-terminated line is a complete statement."""

    def extract(
        self, source: SourceCode, *, final: bool
    ) -> tuple[list[SourceCode], SourceCode]:
        lines = source.split("\n")
        if final:
            stmts = [l for l in lines if l.strip()]
            return stmts, ""
        # Non-final: only lines that end with \n are ready
        complete = lines[:-1]
        leftover = lines[-1]
        stmts = [l for l in complete if l.strip()]
        return stmts, leftover


class FailingExtractor:
    """Extractor that raises SyntaxError on first call."""

    def extract(
        self, source: SourceCode, *, final: bool
    ) -> tuple[list[SourceCode], SourceCode]:
        raise SyntaxError("bad syntax")


class EchoExecutor:
    """Executor that echoes back the source code as output."""

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        return ExecutionResult(success=True, output=f"echo:{source_code.strip()}")


class FailExecutor:
    """Executor that always fails."""

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        return ExecutionResult(success=False, error="exec failed")


class SlowExecutor:
    """Executor that takes a non-trivial amount of time."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.calls: list[str] = []

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        self.calls.append(source_code)
        await asyncio.sleep(self.delay)
        return ExecutionResult(success=True, output=f"slow:{source_code.strip()}")


# ── push / result basics ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_push_and_result_single_statement():
    session = Session(extractor=EchoExtractor(), executor=EchoExecutor())
    session.push("print('hi')\n")
    out = await session.result()
    assert out == "echo:print('hi')"


@pytest.mark.asyncio
async def test_result_concatenates_multiple_statements_in_order():
    slow = SlowExecutor(delay=0.02)
    session = Session(extractor=EchoExtractor(), executor=slow)
    session.push("a\n")
    session.push("b\n")
    session.push("c\n")
    out = await session.result()
    assert out == "slow:aslow:bslow:c"
    assert slow.calls == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_push_is_nonblocking():
    """push() should return before the background task completes."""
    slow = SlowExecutor(delay=0.1)
    session = Session(extractor=EchoExtractor(), executor=slow)
    t0 = asyncio.get_event_loop().time()
    session.push("x\n")
    elapsed = asyncio.get_event_loop().time() - t0
    assert elapsed < 0.05, "push() should return immediately"
    await session.result()  # drain


@pytest.mark.asyncio
async def test_result_empty_when_nothing_pushed():
    session = Session(extractor=EchoExtractor(), executor=EchoExecutor())
    out = await session.result()
    assert out == ""


# ── error handling ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_syntax_error_captured_synchronously_in_push():
    session = Session(extractor=FailingExtractor(), executor=EchoExecutor())
    session.push("bad code")
    # error is observable immediately — no await needed
    assert session.has_error
    assert isinstance(session.error, SyntaxError)


@pytest.mark.asyncio
async def test_result_raises_syntax_error():
    session = Session(extractor=FailingExtractor(), executor=EchoExecutor())
    session.push("bad code")
    with pytest.raises(SyntaxError):
        await session.result()


@pytest.mark.asyncio
async def test_runtime_error_captured_in_task_and_raised_by_result():
    session = Session(extractor=EchoExtractor(), executor=FailExecutor())
    session.push("x\n")
    assert not session.has_error  # not yet set — task is async
    with pytest.raises(RuntimeError, match="exec failed"):
        await session.result()
    assert session.has_error


@pytest.mark.asyncio
async def test_push_after_error_is_no_op():
    session = Session(extractor=EchoExtractor(), executor=FailExecutor())
    session.push("x\n")
    with pytest.raises(RuntimeError):
        await session.result()
    # subsequent push should not enqueue more work
    session.push("y\n")
    assert session.has_error  # still in error state until reset


# ── subsequent statements do not run after error ───────────────────────────────

@pytest.mark.asyncio
async def test_subsequent_statements_not_executed_after_runtime_error():
    executed: list[str] = []

    class TrackingFailFirstExecutor:
        def __init__(self):
            self.n = 0

        async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
            self.n += 1
            if self.n == 1:
                return ExecutionResult(success=False, error="first fails")
            executed.append(source_code)
            return ExecutionResult(success=True, output=source_code)

    executor = TrackingFailFirstExecutor()
    session = Session(extractor=EchoExtractor(), executor=executor)
    session.push("a\n")
    session.push("b\n")
    session.push("c\n")
    with pytest.raises(RuntimeError):
        await session.result()
    # b and c should not have been executed
    assert executed == []


# ── reset ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reset_clears_error_and_output():
    session = Session(extractor=EchoExtractor(), executor=FailExecutor())
    session.push("x\n")
    with pytest.raises(RuntimeError):
        await session.result()
    session.reset()
    assert not session.has_error
    assert session.error is None
    assert session.buffer == ""


@pytest.mark.asyncio
async def test_reset_preserves_executor_repl_state():
    """Executor keeps its own state across reset; Session only resets its own buffers."""
    calls: list[str] = []

    class StatefulExecutor:
        async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
            calls.append(source_code.strip())
            return ExecutionResult(success=True, output=source_code)

    executor = StatefulExecutor()
    session = Session(extractor=EchoExtractor(), executor=executor)
    session.push("first\n")
    await session.result()
    session.reset()
    session.push("second\n")
    await session.result()
    assert calls == ["first", "second"]


@pytest.mark.asyncio
async def test_session_works_after_reset():
    session = Session(extractor=EchoExtractor(), executor=EchoExecutor())
    session.push("a\n")
    out1 = await session.result()
    session.reset()
    session.push("b\n")
    out2 = await session.result()
    assert out1 == "echo:a"
    assert out2 == "echo:b"


# ── executor injection (proves core is agnostic) ───────────────────────────────

@pytest.mark.asyncio
async def test_fake_executor_drives_session_end_to_end():
    """Any BaseExecutor impl can be plugged in — proves core is executor-agnostic."""

    class UppercaseExecutor:
        async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
            return ExecutionResult(success=True, output=source_code.upper())

    session = Session(extractor=EchoExtractor(), executor=UppercaseExecutor())
    session.push("hello\n")
    out = await session.result()
    assert out == "HELLO"
