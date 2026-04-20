"""Tests for InProcPythonExecutor."""

import asyncio
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
async def test_locals_consistent_after_timeout():
    """_locals is consistent after a timed-out call: no concurrent mutations."""
    exec_ = InProcPythonExecutor(timeout=0.1)
    # Times out at 0.1s; zombie continues and sets x = 99 after 0.2s.
    await exec_.aexecute("import time; time.sleep(0.2); x = 99")
    # Wait for the zombie to finish before submitting the next job.
    await asyncio.sleep(0.3)
    # Thread is idle now; this job runs immediately and must see x == 99.
    result = await exec_.aexecute("print(x)")
    assert result.success
    assert result.output == "99\n"
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
