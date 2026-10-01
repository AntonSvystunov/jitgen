import asyncio
import warnings
from collections.abc import Mapping
from typing import Any

from jitgen.base import ExecutionResult, SourceCode
from jitgen.executors.base import ExecutorBase
from quickjs_rs import (
    ConcurrentEvalError,
    Context,
    DeadlockError,
    HostCancellationError,
    JSError,
    MarshalError,
    MemoryLimitError,
    Runtime,
    SourceTransform,
    ThreadWorker,
)
from quickjs_rs import (
    TimeoutError as QuickJsTimeoutError,
)

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024
_DEFAULT_MAX_CONSOLE_CHARS = 100_000


def _describe(exc: BaseException) -> str:
    """Render `exc` as `"ConcurrentEvalError: ..."`, falling back to the class name.

    Mirrors `InProcPythonExecutor`'s `_describe` — `str(exc)` alone can be empty
    for some `quickjs_rs` exceptions, which would otherwise report a blank error.

    Args:
        exc: The exception to describe.

    Returns:
        `exc`'s type name, followed by `": {message}"` when it has one.
    """
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class _ConsoleBuffer:
    """Bounded accumulator for JS `console.*` output, drained once per `aexecute`.

    Bounded because a runaway `console.log` in a loop must not exhaust host
    memory — there is no OS-level stdout stream inside the WASM sandbox to
    apply backpressure to, unlike `InProcPythonExecutor`'s `redirect_stdout`.
    """

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars
        self._parts: list[str] = []
        self._length = 0
        self._truncated = False

    def append(self, text: str) -> None:
        if self._truncated:
            return
        remaining = self._max_chars - self._length
        if remaining <= 0:
            self._truncated = True
            return
        if len(text) > remaining:
            text = text[:remaining]
            self._truncated = True
        self._parts.append(text)
        self._length += len(text)

    def drain(self) -> str | None:
        if not self._parts:
            return None
        output = "".join(self._parts)
        self._parts.clear()
        self._length = 0
        self._truncated = False
        return output


class QuickJsExecutor(ExecutorBase):
    """Runs JavaScript statements against a persistent QuickJS context.

    A single `quickjs_rs.Context` (backed by a `Runtime`) persists across
    calls, giving REPL-style state (P5) — top-level `var`/`let`/`const`/
    `function` declarations from one statement are visible to the next.
    `Context`/`Runtime` objects are `!Send`, so all operations on them are
    marshaled onto one dedicated `quickjs_rs.ThreadWorker` thread.

    Timeouts are enforced natively by the embedded QuickJS engine's own
    interrupt handler (the `timeout=` passed to each `eval_handle_async`
    call), not by injecting an exception into a worker thread the way
    `InProcPythonExecutor` does — CPython has no interrupt primitive for a
    call already running inside a native/WASM extension.

    There is, however, no *cancellation* primitive at all in `quickjs_rs`'s
    current API (confirmed empirically, not assumed: `Runtime` and `Context`
    expose no `interrupt`/`cancel`/`set_deadline` method). Cancelling the
    Python-side future wrapping an in-flight call only abandons *our own*
    await — the worker thread keeps running the JS engine synchronously
    until that call's own `timeout` elapses, since QuickJS's WASM execution
    blocks the worker thread's event loop for the call's whole duration and
    cannot be interrupted from outside it. Actually cancelling would be
    worse than a no-op: it would report the statement as cancelled while
    leaving the worker thread still occupied by it, silently blocking every
    later statement until the original timeout eventually fires anyway.
    `acancel` is therefore a documented no-op — this is contract-safe, since
    `Session._acancel` already wraps executor cancellation so a non-cancelable
    executor cannot break `reset()` — and the only real bound on a runaway
    statement is `timeout` itself.

    Known deviations from post-hoc (whole-script) execution, both a
    consequence of dispatching one statement per `eval` call rather than
    running one continuous script (see METHOD.md P4):

    - Top-level `const` is rewritten to `var` (`SourceTransform.
      TOP_LEVEL_CONST_TO_VAR`), so redeclaring a `const` name across two
      separately-dispatched statements silently succeeds here where running
      the whole script at once would throw.
    - JS function-hoisting across statement boundaries does not work: a
      statement calling a function declared later in the same buffer fails
      under incremental dispatch even though it would succeed whole-program.

    Args:
        timeout: Max seconds of JS VM execution time allowed per `aexecute`
            call (wall-clock time spent awaiting async host calls is not
            counted against it — this is `quickjs_rs`'s own accounting).
        memory_limit: Max bytes the underlying QuickJS heap may grow to.
        close_timeout: Max seconds `aclose()` waits for the worker thread to
            stop before abandoning it.
        tools: Names bound as JS global functions at context construction.
        max_console_chars: Max characters retained by the console-output
            buffer before further output is silently dropped.
    """

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        memory_limit: int | None = _DEFAULT_MEMORY_LIMIT,
        close_timeout: float = 5.0,
        tools: Mapping[str, Any] | None = None,
        max_console_chars: int = _DEFAULT_MAX_CONSOLE_CHARS,
    ) -> None:
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.close_timeout = close_timeout
        self.tools: dict[str, Any] = dict(tools or {})

        self._worker = ThreadWorker(name="jitgen-quickjs-executor")
        self._runtime: Runtime | None = None
        self._ctx: Context | None = None
        self._init_lock = asyncio.Lock()
        self._console = _ConsoleBuffer(max_console_chars)

    async def _ensure_context(self) -> Context:
        """Build the persistent `Runtime`/`Context` pair on first use.

        Lazy because `Runtime`/`Context` construction must happen on the
        worker thread (they are `!Send`), which means it must be async even
        though the constructors themselves are synchronous — it cannot run
        eagerly in `__init__`. Guarded by a lock so two concurrent first
        calls to `aexecute` cannot each build their own context.
        """
        if self._ctx is not None:
            return self._ctx
        async with self._init_lock:
            if self._ctx is not None:
                return self._ctx

            async def _build() -> tuple[Runtime, Context]:
                runtime = Runtime(
                    memory_limit=self.memory_limit,
                    transform_flags=SourceTransform.TOP_LEVEL_CONST_TO_VAR,
                )
                ctx = runtime.new_context(timeout=self.timeout)
                self._install_console(ctx)
                self._install_tools(ctx)
                return runtime, ctx

            self._runtime, self._ctx = await self._worker.run_async(_build())
            return self._ctx

    def _install_console(self, ctx: Context) -> None:
        """Wire `globalThis.console` to host functions that fill `self._console`.

        Registered on the worker thread as part of context construction —
        `Context.register` touches the `!Send` context and must not be
        called from any other thread.
        """

        def _record(*args: Any) -> None:
            self._console.append(" ".join(str(arg) for arg in args) + "\n")

        ctx.register("__jitgen_console_log", _record)
        ctx.register("__jitgen_console_warn", _record)
        ctx.register("__jitgen_console_error", _record)
        # The trailing `undefined;` matters: without it, `eval`'s attempt to
        # marshal the assigned object's value (which contains function
        # properties) back to Python raises `MarshalError`.
        ctx.eval(
            "globalThis.console = {"
            " log: __jitgen_console_log,"
            " warn: __jitgen_console_warn,"
            " error: __jitgen_console_error,"
            "};"
            "undefined;"
        )

    def _install_tools(self, ctx: Context) -> None:
        for name, tool in self.tools.items():
            ctx.register(name, tool)

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        """Run `source_code` in the persistent context and report what happened.

        Uses `eval_handle_async` rather than `eval`/`eval_async` so a
        statement whose completion value happens to be a function (e.g. a
        bare `foo;` where `foo` is a declared function) never fails with
        `MarshalError` merely because its return value can't cross back to
        Python — `jitgen`'s `ExecutionResult.output` only ever carries
        captured console text, never the statement's value, so the handle is
        disposed unread.

        Args:
            source_code: One dispatched top-level JS statement.

        Returns:
            The outcome of running `source_code`.
        """
        ctx = await self._ensure_context()
        try:
            handle = await self._worker.run_async(
                ctx.eval_handle_async(source_code, timeout=self.timeout)
            )
        except QuickJsTimeoutError as exc:
            return ExecutionResult(
                success=False,
                output=self._console.drain(),
                error=str(exc) or f"Execution timed out after {self.timeout}s",
                has_timed_out=True,
            )
        except JSError as exc:
            return ExecutionResult(
                success=False, output=self._console.drain(), error=str(exc)
            )
        except MemoryLimitError as exc:
            return ExecutionResult(
                success=False,
                output=self._console.drain(),
                error=f"OutOfMemory: {exc}",
            )
        except (
            ConcurrentEvalError,
            DeadlockError,
            MarshalError,
            HostCancellationError,
        ) as exc:
            return ExecutionResult(
                success=False, output=self._console.drain(), error=_describe(exc)
            )
        except Exception as exc:  # noqa: BLE001  # a tool raised: the statement failed
            # `quickjs_rs` re-raises a host function's exception as-is. Report it
            # like any other failure, draining the console so the output this
            # statement printed doesn't surface in the next one.
            return ExecutionResult(
                success=False, output=self._console.drain(), error=_describe(exc)
            )
        else:
            handle.dispose()
            return ExecutionResult(success=True, output=self._console.drain())

    async def acancel(self) -> None:
        """No-op: `quickjs_rs` exposes no primitive to interrupt a running call.

        See the class docstring for why attempting one via Python-side task
        cancellation would be actively worse than doing nothing. The only
        real bound on a runaway statement is `timeout`.
        """

    async def aclose(self) -> None:
        """Release the context, runtime, and worker thread.

        Ordered context → runtime → worker: the first two are `!Send` and
        must be torn down on the worker thread; the worker itself is stopped
        last so it is available to run that teardown coroutine.

        A statement already running past `close_timeout` cannot be sped up
        (see the class docstring), so — mirroring `InProcPythonExecutor.
        aclose`'s abandonment fallback — the worker thread is abandoned
        rather than waited on indefinitely; it is a daemon thread, so this
        does not keep the process alive, but the runaway statement keeps the
        thread busy until its own `timeout` elapses regardless.
        """
        ctx, runtime = self._ctx, self._runtime
        if ctx is not None or runtime is not None:

            async def _teardown() -> None:
                if ctx is not None:
                    ctx.close()
                if runtime is not None:
                    runtime.close()

            try:
                await asyncio.wait_for(
                    self._worker.run_async(_teardown()), self.close_timeout
                )
            except Exception:  # noqa: BLE001, S110
                pass  # best-effort; the worker close below abandons it anyway
            self._ctx = None
            self._runtime = None

        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._worker.close), self.close_timeout
            )
        except TimeoutError:
            warnings.warn(
                "QuickJsExecutor: worker thread did not stop within "
                f"{self.close_timeout}s and was abandoned — a statement is "
                "still running inside the QuickJS engine and can only stop "
                "once its own `timeout` elapses.",
                RuntimeWarning,
                stacklevel=2,
            )
