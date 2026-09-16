import asyncio
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from jitgen.executors.python import InProcPythonExecutor


@pytest.fixture
async def make_executor() -> AsyncIterator[Callable[..., InProcPythonExecutor]]:
    # Factory rather than a fixed instance: several tests need their own
    # timeout/close_timeout/tools. Every instance created through it is
    # closed on teardown so no worker thread outlives its test.
    created: list[InProcPythonExecutor] = []

    def factory(**kwargs: Any) -> InProcPythonExecutor:
        instance = InProcPythonExecutor(**kwargs)
        created.append(instance)
        return instance

    try:
        yield factory
    finally:
        for instance in created:
            await instance.aclose()


@pytest.fixture
def executor(
    make_executor: Callable[..., InProcPythonExecutor],
) -> InProcPythonExecutor:
    return make_executor()


async def test_hello_world(executor: InProcPythonExecutor):
    result = await executor.aexecute("print('Hello, World!')")
    assert result.success
    assert result.output == "Hello, World!\n"
    assert result.error is None
    assert not result.has_timed_out


async def test_runtime_error(executor: InProcPythonExecutor):
    result = await executor.aexecute("print(undefined_variable)")
    assert not result.success
    assert result.error is not None
    assert "undefined_variable" in result.error
    assert not result.has_timed_out


async def test_repl_state_persists(executor: InProcPythonExecutor):
    await executor.aexecute("x = 42")
    result = await executor.aexecute("print(x)")
    assert result.success
    assert result.output == "42\n"


async def test_timeout_configured_on_executor():
    # Constructed directly, not via make_executor: time.sleep(10) is
    # uninterruptible, so closing here would just burn close_timeout waiting
    # it out. The daemon thread is reaped when the process exits.
    slow_exec = InProcPythonExecutor(timeout=0.1)
    result = await slow_exec.aexecute("import time; time.sleep(10)")
    assert not result.success
    assert result.has_timed_out


async def test_empty_code_succeeds(executor: InProcPythonExecutor):
    result = await executor.aexecute("")
    assert result.success


async def test_output_is_none_when_no_print(executor: InProcPythonExecutor):
    result = await executor.aexecute("x = 1")
    assert result.success
    assert result.output is None


async def test_multiple_print_statements(executor: InProcPythonExecutor):
    result = await executor.aexecute("print('a'); print('b'); print('c')")
    assert result.success
    assert result.output == "a\nb\nc\n"


async def test_exception_captured_in_result(executor: InProcPythonExecutor):
    result = await executor.aexecute("raise ValueError('boom')")
    assert not result.success
    assert result.error is not None
    assert "boom" in result.error


async def test_tools_registered_at_construction(
    make_executor: Callable[..., InProcPythonExecutor],
):
    exec_ = make_executor(tools={"add": lambda a, b: a + b})
    result = await exec_.aexecute("print(add(2, 3))")
    assert result.success
    assert result.output == "5\n"


async def test_register_tool_after_construction(executor: InProcPythonExecutor):
    executor.register_tool("greet", lambda name: f"hi {name}")
    result = await executor.aexecute("print(greet('world'))")
    assert result.success
    assert result.output == "hi world\n"


async def test_tools_isolated_between_executors(
    make_executor: Callable[..., InProcPythonExecutor],
):
    make_executor(tools={"secret": lambda: 42})
    other = make_executor()
    result = await other.aexecute("print('secret' in dir())")
    assert result.success
    assert result.output == "False\n"


async def test_tool_exception_captured(executor: InProcPythonExecutor):
    executor.register_tool("bad", lambda: (_ for _ in ()).throw(RuntimeError("oops")))
    result = await executor.aexecute("bad()")
    assert not result.success
    assert result.error is not None
    assert "oops" in result.error


async def test_success_after_failure_is_independent(executor: InProcPythonExecutor):
    fail = await executor.aexecute("raise ValueError('first')")
    assert not fail.success
    ok = await executor.aexecute("print('ok')")
    assert ok.success
    assert ok.output == "ok\n"


async def test_open_tool_can_read_file(
    make_executor: Callable[..., InProcPythonExecutor], tmp_path: Path
):
    f = tmp_path / "data.txt"
    f.write_text("hello", encoding="utf-8")
    exec_ = make_executor(tools={"open": open})
    result = await exec_.aexecute(
        f"with open({str(f)!r}, encoding='utf-8') as h:\n    print(h.read())"
    )
    assert result.success
    assert result.output == "hello\n"


# ── single-worker-thread serialization & cancellation ─────────────────────────


async def test_timeout_still_prompt_on_caller_side():
    """aexecute returns within ~timeout even though the thread keeps running."""
    # See test_timeout_configured_on_executor: not routed through
    # make_executor for the same reason (uninterruptible 10s sleep).
    exec_ = InProcPythonExecutor(timeout=0.1)
    t0 = time.monotonic()
    result = await exec_.aexecute("import time; time.sleep(10)")
    elapsed = time.monotonic() - t0
    assert result.has_timed_out
    assert not result.success
    assert elapsed < 1.0, f"aexecute blocked for {elapsed:.2f}s, expected ~0.1s"


async def test_timed_out_statement_does_not_keep_mutating_state(
    make_executor: Callable[..., InProcPythonExecutor],
):
    """A timed-out statement is aborted, not left to finish in the background.

    It used to run to completion on the worker thread, so state written *after*
    the caller had already been told the call timed out still landed in the REPL
    namespace — a later statement would then observe a value from a statement
    that had officially failed.  The trailing assignment must not take effect.
    """
    exec_ = make_executor(timeout=0.1)
    # Times out at 0.1s.  The interrupt is delivered once time.sleep returns,
    # before the assignment's bytecode runs.
    result = await exec_.aexecute("import time; time.sleep(0.2); x = 99")
    assert result.has_timed_out

    await asyncio.sleep(0.4)  # give the interrupted job time to unwind
    probe = await exec_.aexecute("print('x' in dir())")
    assert probe.success
    assert probe.output == "False\n"


async def test_locals_consistent_after_timeout(
    make_executor: Callable[..., InProcPythonExecutor],
):
    """_locals stays usable after a timed-out call: no concurrent mutations."""
    exec_ = make_executor(timeout=0.1)
    await exec_.aexecute("before = 1")
    await exec_.aexecute("import time; time.sleep(0.2)")
    await asyncio.sleep(0.4)

    result = await exec_.aexecute("print(before)")
    assert result.success
    assert result.output == "1\n"


async def test_system_exit_does_not_kill_worker(executor: InProcPythonExecutor):
    """sys.exit() is caught; subsequent aexecute still works."""
    result_exit = await executor.aexecute("import sys; sys.exit(0)")
    assert not result_exit.success
    assert result_exit.error is not None
    assert "SystemExit" in result_exit.error
    result_ok = await executor.aexecute("print('alive')")
    assert result_ok.success
    assert result_ok.output == "alive\n"


async def test_acancel_is_safe_before_any_work(executor: InProcPythonExecutor):
    """acancel() before any execution must not raise."""
    await executor.acancel()
    result = await executor.aexecute("print(1)")
    assert result.success


async def test_aclose_joins_worker_thread(executor: InProcPythonExecutor):
    """After aclose(), the worker thread must be dead."""
    await executor.aexecute("x = 1")
    await executor.aclose()
    assert not executor._thread.is_alive()


# ── runaway code must not wedge the executor ──────────────────────────────────


async def test_infinite_loop_is_interrupted_at_timeout(
    make_executor: Callable[..., InProcPythonExecutor],
):
    """A ``while True:`` must stop when its budget runs out.

    Without interruption the worker thread spins for the life of the process,
    burning a core and holding the process-wide stdout redirect.
    """
    exec_ = make_executor(timeout=0.5)
    result = await exec_.aexecute("while True:\n    pass")
    assert result.has_timed_out
    assert not result.success


async def test_executor_still_usable_after_a_runaway_job(
    make_executor: Callable[..., InProcPythonExecutor],
):
    exec_ = make_executor(timeout=0.5)
    await exec_.aexecute("while True:\n    pass")

    result = await exec_.aexecute("print(6 * 7)")
    assert result.success
    assert result.output == "42\n"


async def test_repl_state_survives_an_interrupt(
    make_executor: Callable[..., InProcPythonExecutor],
):
    exec_ = make_executor(timeout=0.5)
    await exec_.aexecute("kept = 'before'")
    await exec_.aexecute("while True:\n    pass")

    result = await exec_.aexecute("print(kept)")
    assert result.output == "before\n"


async def test_std_streams_restored_after_a_runaway_job(
    make_executor: Callable[..., InProcPythonExecutor],
):
    """The redirect is process-wide, so a job that never unwinds silently eats
    every later write in the whole process — progress bars, logs, tracebacks."""
    # Snapshot rather than compare against sys.__stdout__: pytest's capture has
    # already swapped the streams, and what matters is that we hand back
    # whatever was in place before the job ran.
    stdout_before, stderr_before = sys.stdout, sys.stderr
    exec_ = make_executor(timeout=0.5)
    await exec_.aexecute("while True:\n    pass")

    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before


async def test_aclose_is_bounded_even_while_code_is_still_running(
    make_executor: Callable[..., InProcPythonExecutor],
):
    """Closing mid-execution must interrupt rather than wait it out."""
    exec_ = make_executor(timeout=30.0, close_timeout=2.0)
    task = asyncio.create_task(exec_.aexecute("while True:\n    pass"))
    await asyncio.sleep(0.2)  # let the job reach the worker thread

    await asyncio.wait_for(exec_.aclose(), timeout=5.0)
    task.cancel()


async def test_acancel_interrupts_running_code(
    make_executor: Callable[..., InProcPythonExecutor],
):
    exec_ = make_executor(timeout=30.0)
    task = asyncio.create_task(exec_.aexecute("while True:\n    pass"))
    await asyncio.sleep(0.2)

    await exec_.acancel()
    result = await asyncio.wait_for(task, timeout=5.0)
    assert result.has_cancelled
    assert not result.success


async def test_interrupt_does_not_leak_into_the_next_statement(
    make_executor: Callable[..., InProcPythonExecutor],
):
    """An interrupt landing just as its job finishes must not abort the next one."""
    exec_ = make_executor(timeout=30.0)
    for _ in range(20):
        await exec_.aexecute("y = 1")
        await exec_.acancel()  # races against the job that just completed

    result = await exec_.aexecute("print('still here')")
    assert result.success
    assert result.output == "still here\n"
