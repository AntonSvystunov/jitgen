import time
from collections.abc import Callable
from typing import Any

from jitgen_js.executors.quickjs import QuickJsExecutor


async def test_hello_world(executor: QuickJsExecutor):
    result = await executor.aexecute("console.log('Hello, World!');")
    assert result.success
    assert result.output == "Hello, World!\n"
    assert result.error is None
    assert not result.has_timed_out


async def test_runtime_error(executor: QuickJsExecutor):
    result = await executor.aexecute("undefinedFn();")
    assert not result.success
    assert result.error is not None
    assert "undefinedFn" in result.error
    assert not result.has_timed_out


async def test_repl_state_persists_across_calls(executor: QuickJsExecutor):
    await executor.aexecute("var x = 42;")
    result = await executor.aexecute("console.log(x);")
    assert result.success
    assert result.output == "42\n"


async def test_top_level_const_can_be_redeclared_across_calls(
    executor: QuickJsExecutor,
):
    # Consequence of SourceTransform.TOP_LEVEL_CONST_TO_VAR (documented on
    # QuickJsExecutor as a deliberate P4 deviation): whole-program execution
    # of `const x = 1; const x = 2;` would throw, but under incremental
    # per-statement dispatch it must not.
    first = await executor.aexecute("const x = 1;")
    second = await executor.aexecute("const x = 2; console.log(x);")
    assert first.success
    assert second.success
    assert second.output == "2\n"


async def test_function_valued_statement_does_not_fail_to_marshal(
    executor: QuickJsExecutor,
):
    # A statement whose completion value is a function must not fail with
    # MarshalError merely because that value can't cross back to Python —
    # ExecutionResult never carries the statement's value, only console
    # output, so this must succeed.
    await executor.aexecute("function f(){ return 1; }")
    result = await executor.aexecute("f;")
    assert result.success


async def test_timeout_returns_promptly_instead_of_blocking(
    make_executor: Callable[..., QuickJsExecutor],
):
    executor = make_executor(timeout=0.5)
    start = time.perf_counter()
    result = await executor.aexecute("while (true) {}")
    elapsed = time.perf_counter() - start
    assert not result.success
    assert result.has_timed_out
    # A generous bound well under what an un-interrupted infinite loop would
    # take — this is the test that would catch a regression to a
    # cooperative-only cancellation that can't stop a tight, non-yielding
    # JS loop the way quickjs_rs's native VM interrupt can.
    assert elapsed < 5.0


async def test_context_is_still_usable_after_a_timeout(
    make_executor: Callable[..., QuickJsExecutor],
):
    executor = make_executor(timeout=0.5)
    await executor.aexecute("while (true) {}")
    result = await executor.aexecute("console.log('still alive');")
    assert result.success
    assert result.output == "still alive\n"


async def test_acancel_is_a_safe_no_op(make_executor: Callable[..., QuickJsExecutor]):
    # `quickjs_rs` exposes no primitive to interrupt a running call (see
    # QuickJsExecutor's docstring): cancelling the Python-side future would
    # only abandon our own await while the worker thread kept running the
    # statement until its own timeout — worse than doing nothing, since it
    # would misreport an still-running statement as cancelled. `acancel` is
    # therefore a deliberate no-op; this only asserts it never raises,
    # narrowly, so it does not silently pass if real cancellation is added
    # later without updating this test.
    executor = make_executor(timeout=0.5)
    await executor.acancel()  # before any work — must not raise
    result = await executor.aexecute("console.log('unaffected by acancel');")
    assert result.success
    assert result.output == "unaffected by acancel\n"


async def test_aclose_is_idempotent(executor: QuickJsExecutor):
    await executor.aclose()
    await executor.aclose()  # must not raise


async def test_tools_are_bound_as_js_globals_at_construction(
    make_executor: Callable[..., QuickJsExecutor],
):
    calls: list[Any] = []

    def add(a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    executor = make_executor(tools={"add": add})
    result = await executor.aexecute("console.log(add(2, 3));")
    assert result.success
    assert result.output == "5\n"
    assert calls == [(2, 3)]


async def test_a_raising_tool_fails_the_statement_instead_of_raising(
    make_executor: Callable[..., QuickJsExecutor],
):
    async def lookup(**kwargs: Any) -> str:
        msg = "lookup() missing required argument(s): case_id"
        raise TypeError(msg)

    executor = make_executor(tools={"lookup": lookup})
    result = await executor.aexecute('console.log("before"); await lookup();')

    assert not result.success
    assert result.error == "TypeError: lookup() missing required argument(s): case_id"
    assert result.output == "before\n"


async def test_output_of_a_failed_tool_call_does_not_leak_into_the_next_statement(
    make_executor: Callable[..., QuickJsExecutor],
):
    def fail() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    executor = make_executor(tools={"fail": fail})
    await executor.aexecute('console.log("before"); fail();')
    result = await executor.aexecute('console.log("next");')

    assert result.success
    assert result.output == "next\n"
