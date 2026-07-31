"""Tests for InProcPythonExecutor."""

import asyncio
import sys
import time

import pytest

from jitgen.executors.python import InProcPythonExecutor


@pytest.fixture()
def executor():
    return InProcPythonExecutor()


@pytest.mark.asyncio
async def test_hello_world(executor: InProcPythonExecutor):
    result = await executor.aexecute("print('Hello, World!')")
    assert result.success
    assert result.output == "Hello, World!\n"
    assert result.error is None
    assert not result.has_timed_out


@pytest.mark.asyncio
async def test_runtime_error(executor: InProcPythonExecutor):
    result = await executor.aexecute("print(undefined_variable)")
    assert not result.success
    assert result.error is not None
    assert "undefined_variable" in result.error
    assert not result.has_timed_out


@pytest.mark.asyncio
async def test_repl_state_persists(executor: InProcPythonExecutor):
    await executor.aexecute("x = 42")
    result = await executor.aexecute("print(x)")
    assert result.success
    assert result.output == "42\n"


@pytest.mark.asyncio
async def test_locals_dict_populated(executor: InProcPythonExecutor):
    await executor.aexecute("a = 10")
    assert executor._locals["a"] == 10  # noqa: SLF001


@pytest.mark.asyncio
async def test_timeout_configured_on_executor():
    slow_exec = InProcPythonExecutor(timeout=0.1)
    result = await slow_exec.aexecute("import time; time.sleep(10)")
    assert not result.success
    assert result.has_timed_out


@pytest.mark.asyncio
async def test_no_timeout_by_default_for_fast_code(executor: InProcPythonExecutor):
    result = await executor.aexecute("print('fast')")
    assert result.success
    assert not result.has_timed_out


@pytest.mark.asyncio
async def test_empty_code_succeeds(executor: InProcPythonExecutor):
    result = await executor.aexecute("")
    assert result.success


@pytest.mark.asyncio
async def test_output_is_none_when_no_print(executor: InProcPythonExecutor):
    result = await executor.aexecute("x = 1")
    assert result.success
    assert result.output is None


@pytest.mark.asyncio
async def test_multiple_print_statements(executor: InProcPythonExecutor):
    result = await executor.aexecute("print('a'); print('b'); print('c')")
    assert result.success
    assert result.output == "a\nb\nc\n"


@pytest.mark.asyncio
async def test_exception_captured_in_result(executor: InProcPythonExecutor):
    result = await executor.aexecute("raise ValueError('boom')")
    assert not result.success
    assert result.error is not None
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_tools_registered_at_construction():
    exec_ = InProcPythonExecutor(tools={"add": lambda a, b: a + b})
    result = await exec_.aexecute("print(add(2, 3))")
    assert result.success
    assert result.output == "5\n"


@pytest.mark.asyncio
async def test_register_tool_after_construction(executor: InProcPythonExecutor):
    executor.register_tool("greet", lambda name: f"hi {name}")
    result = await executor.aexecute("print(greet('world'))")
    assert result.success
    assert result.output == "hi world\n"


@pytest.mark.asyncio
async def test_tools_isolated_between_executors():
    e1 = InProcPythonExecutor(tools={"secret": lambda: 42})
    e2 = InProcPythonExecutor()
    result = await e2.aexecute("print('secret' in dir())")
    assert result.success
    assert result.output == "False\n"


@pytest.mark.asyncio
async def test_tool_exception_captured(executor: InProcPythonExecutor):
    executor.register_tool("bad", lambda: (_ for _ in ()).throw(RuntimeError("oops")))
    result = await executor.aexecute("bad()")
    assert not result.success
    assert "oops" in result.error


@pytest.mark.asyncio
async def test_success_after_failure_is_independent(executor: InProcPythonExecutor):
    fail = await executor.aexecute("raise ValueError('first')")
    assert not fail.success
    ok = await executor.aexecute("print('ok')")
    assert ok.success
    assert ok.output == "ok\n"


@pytest.mark.asyncio
async def test_open_tool_can_read_file(tmp_path: pytest.fixture):
    f = tmp_path / "data.txt"
    f.write_text("hello", encoding="utf-8")
    exec_ = InProcPythonExecutor(tools={"open": open})
    result = await exec_.aexecute(
        f"with open({str(f)!r}, encoding='utf-8') as h:\n    print(h.read())"
    )
    assert result.success
    assert result.output == "hello\n"


# ── single-worker-thread serialization & cancellation ─────────────────────────

@pytest.mark.asyncio
async def test_timeout_still_prompt_on_caller_side():
    """aexecute returns within ~timeout even though the thread keeps running."""
    exec_ = InProcPythonExecutor(timeout=0.1)
    t0 = time.monotonic()
    result = await exec_.aexecute("import time; time.sleep(10)")
    elapsed = time.monotonic() - t0
    assert result.has_timed_out
    assert not result.success
    assert elapsed < 1.0, f"aexecute blocked for {elapsed:.2f}s, expected ~0.1s"
    # Do not call aclose() here — the zombie thread is still sleeping (10s).
    # Daemon threads are killed when the process exits; no join needed.


@pytest.mark.asyncio
async def test_timed_out_statement_does_not_keep_mutating_state():
    """A timed-out statement is aborted, not left to finish in the background.

    It used to run to completion on the worker thread, so state written *after*
    the caller had already been told the call timed out still landed in the REPL
    namespace — a later statement would then observe a value from a statement
    that had officially failed.  The trailing assignment must not take effect.
    """
    exec_ = InProcPythonExecutor(timeout=0.1)
    # Times out at 0.1s.  The interrupt is delivered once time.sleep returns,
    # before the assignment's bytecode runs.
    result = await exec_.aexecute("import time; time.sleep(0.2); x = 99")
    assert result.has_timed_out

    await asyncio.sleep(0.4)  # give the interrupted job time to unwind
    probe = await exec_.aexecute("print('x' in dir())")
    assert probe.success
    assert probe.output == "False\n"
    await exec_.aclose()


@pytest.mark.asyncio
async def test_locals_consistent_after_timeout():
    """_locals stays usable after a timed-out call: no concurrent mutations."""
    exec_ = InProcPythonExecutor(timeout=0.1)
    await exec_.aexecute("before = 1")
    await exec_.aexecute("import time; time.sleep(0.2)")
    await asyncio.sleep(0.4)

    result = await exec_.aexecute("print(before)")
    assert result.success
    assert result.output == "1\n"
    await exec_.aclose()


@pytest.mark.asyncio
async def test_system_exit_does_not_kill_worker(executor: InProcPythonExecutor):
    """sys.exit() is caught; subsequent aexecute still works."""
    result_exit = await executor.aexecute("import sys; sys.exit(0)")
    assert not result_exit.success
    assert "SystemExit" in result_exit.error
    result_ok = await executor.aexecute("print('alive')")
    assert result_ok.success
    assert result_ok.output == "alive\n"


@pytest.mark.asyncio
async def test_acancel_is_safe_before_any_work():
    """acancel() before any execution must not raise."""
    exec_ = InProcPythonExecutor()
    await exec_.acancel()
    result = await exec_.aexecute("print(1)")
    assert result.success
    await exec_.aclose()


@pytest.mark.asyncio
async def test_aclose_joins_worker_thread():
    """After aclose(), the worker thread must be dead."""
    exec_ = InProcPythonExecutor()
    await exec_.aexecute("x = 1")
    await exec_.aclose()
    assert not exec_._thread.is_alive()  # noqa: SLF001


# ── runaway code must not wedge the executor ──────────────────────────────────

@pytest.mark.asyncio
async def test_infinite_loop_is_interrupted_at_timeout():
    """A ``while True:`` must stop when its budget runs out.

    Without interruption the worker thread spins for the life of the process,
    burning a core and holding the process-wide stdout redirect.
    """
    executor = InProcPythonExecutor(timeout=0.5)
    result = await executor.aexecute("while True:\n    pass")
    assert result.has_timed_out
    assert not result.success
    await executor.aclose()


@pytest.mark.asyncio
async def test_executor_still_usable_after_a_runaway_job():
    executor = InProcPythonExecutor(timeout=0.5)
    await executor.aexecute("while True:\n    pass")

    result = await executor.aexecute("print(6 * 7)")
    assert result.success
    assert result.output == "42\n"
    await executor.aclose()


@pytest.mark.asyncio
async def test_repl_state_survives_an_interrupt():
    executor = InProcPythonExecutor(timeout=0.5)
    await executor.aexecute("kept = 'before'")
    await executor.aexecute("while True:\n    pass")

    result = await executor.aexecute("print(kept)")
    assert result.output == "before\n"
    await executor.aclose()


@pytest.mark.asyncio
async def test_std_streams_restored_after_a_runaway_job():
    """The redirect is process-wide, so a job that never unwinds silently eats
    every later write in the whole process — progress bars, logs, tracebacks."""
    # Snapshot rather than compare against sys.__stdout__: pytest's capture has
    # already swapped the streams, and what matters is that we hand back
    # whatever was in place before the job ran.
    stdout_before, stderr_before = sys.stdout, sys.stderr
    executor = InProcPythonExecutor(timeout=0.5)
    await executor.aexecute("while True:\n    pass")

    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before
    await executor.aclose()


@pytest.mark.asyncio
async def test_aclose_does_not_hang_on_a_runaway_job():
    """``aclose`` used to join the worker unboundedly, so a runaway loop hung the
    caller forever — the whole evaluation would freeze on teardown."""
    executor = InProcPythonExecutor(timeout=0.5, close_timeout=2.0)
    await executor.aexecute("while True:\n    pass")

    await asyncio.wait_for(executor.aclose(), timeout=5.0)
    assert not executor._thread.is_alive()


@pytest.mark.asyncio
async def test_aclose_is_bounded_even_while_code_is_still_running():
    """Closing mid-execution must interrupt rather than wait it out."""
    executor = InProcPythonExecutor(timeout=30.0, close_timeout=2.0)
    task = asyncio.create_task(executor.aexecute("while True:\n    pass"))
    await asyncio.sleep(0.2)  # let the job reach the worker thread

    await asyncio.wait_for(executor.aclose(), timeout=5.0)
    task.cancel()


@pytest.mark.asyncio
async def test_acancel_interrupts_running_code():
    executor = InProcPythonExecutor(timeout=30.0)
    task = asyncio.create_task(executor.aexecute("while True:\n    pass"))
    await asyncio.sleep(0.2)

    await executor.acancel()
    result = await asyncio.wait_for(task, timeout=5.0)
    assert result.has_cancelled
    assert not result.success
    await executor.aclose()


@pytest.mark.asyncio
async def test_interrupt_does_not_leak_into_the_next_statement():
    """An interrupt landing just as its job finishes must not abort the next one."""
    executor = InProcPythonExecutor(timeout=30.0)
    for _ in range(20):
        await executor.aexecute("y = 1")
        await executor.acancel()  # races against the job that just completed

    result = await executor.aexecute("print('still here')")
    assert result.success
    assert result.output == "still here\n"
    await executor.aclose()
