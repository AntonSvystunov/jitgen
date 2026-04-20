import asyncio
import contextlib
import io
import queue
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import CodeType
from typing import Any  # noqa: ANN401

from pydantic import BaseModel, Field, PrivateAttr

from jitgen_core import ExecutionResult, SourceCode


@dataclass
class _Job:
    compiled_code: CodeType
    stdout: io.StringIO
    stderr: io.StringIO
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[None]


class _ThreadShutdown:
    pass


class InProcPythonExecutor(BaseModel):
    """In-process Python executor with REPL-style persistent state.

    ``timeout`` is owned by the executor instance; callers do not pass it
    per-call.  Instantiate with the desired value once and inject into
    :class:`~jitgen_core.Session`.

    A persistent ``_locals`` dict maintains variable state across executions
    (same semantics as a Python REPL).

    All ``exec()`` calls are serialized through a single dedicated worker
    thread.  When a call times out, the caller returns promptly while the
    thread finishes the in-flight ``exec`` before accepting the next job.
    This guarantees ``_locals`` is never written by two threads simultaneously.

    .. note::
        In-process execution cannot forcibly kill runaway code.  Timed-out
        ``exec`` calls continue on the worker thread until completion.  Use a
        subprocess-based executor if true kill is required.
    """

    timeout: float = Field(
        default=60.0,
        description="Max seconds allowed per aexecute call.",
    )
    tools: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    _locals: dict[str, Any] = PrivateAttr(
        default_factory=lambda: {
            "__name__": "__console__",
            "__doc__": None,
        }
    )
    _thread_queue: queue.Queue[_Job | _ThreadShutdown] = PrivateAttr(default=None)
    _thread: threading.Thread = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.register_tools(self.tools)
        self._thread_queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker_thread_loop,
            daemon=True,
            name="jitgen-inproc-executor",
        )
        self._thread.start()

    def register_tool(self, name: str, tool: Any) -> None:
        self.tools[name] = tool
        self._locals[name] = tool

    def register_tools(self, tools: Mapping[str, Any]) -> None:
        for name, tool in tools.items():
            self.register_tool(name, tool)

    def _worker_thread_loop(self) -> None:
        while True:
            job = self._thread_queue.get()
            if isinstance(job, _ThreadShutdown):
                return
            try:
                with contextlib.redirect_stdout(job.stdout):
                    with contextlib.redirect_stderr(job.stderr):
                        exec(job.compiled_code, self._locals)  # noqa: S102
                job.loop.call_soon_threadsafe(job.future.set_result, None)
            except SystemExit as exc:
                err = RuntimeError(f"SystemExit: {exc.code}")
                job.loop.call_soon_threadsafe(job.future.set_exception, err)
            except BaseException as exc:  # noqa: BLE001
                try:
                    job.loop.call_soon_threadsafe(job.future.set_exception, exc)
                except Exception:  # noqa: BLE001
                    pass

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        """Execute *source_code* asynchronously, returning an :class:`~jitgen_core.ExecutionResult`.

        Submits the compiled code to the dedicated worker thread and awaits
        completion.  On timeout the caller returns immediately while the worker
        thread continues; the next ``aexecute`` call queues behind it,
        ensuring ``_locals`` is never mutated by two threads concurrently.
        """
        stdout_collector = io.StringIO()
        stderr_collector = io.StringIO()
        has_error = False
        has_timed_out = False
        last_exception: BaseException | None = None

        try:
            compiled_code = compile(source_code, "<string>", "exec", optimize=2)
        except Exception as exc:
            return ExecutionResult(success=False, error=str(exc))

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._thread_queue.put_nowait(
            _Job(compiled_code, stdout_collector, stderr_collector, loop, future)
        )

        try:
            # shield prevents wait_for from cancelling the future when the
            # timeout fires — the worker thread keeps its reference and will
            # resolve it when exec() completes.
            await asyncio.wait_for(asyncio.shield(future), self.timeout)
        except asyncio.TimeoutError as exc:
            has_timed_out = True
            has_error = True
            last_exception = exc
        except Exception as exc:
            has_error = True
            last_exception = exc

        return ExecutionResult(
            success=not has_error,
            output=stdout_collector.getvalue() or None,
            error=str(last_exception) if last_exception is not None else None,
            has_timed_out=has_timed_out,
        )

    async def acancel(self) -> None:
        """Best-effort cancellation signal.

        In-process execution cannot be forcibly stopped.  The serialization
        guarantee (single worker thread) ensures the next ``aexecute`` call
        will not run until any currently executing code finishes, so
        ``_locals`` is always in a consistent state.
        """

    async def aclose(self) -> None:
        """Shut down the worker thread cleanly."""
        self._thread_queue.put_nowait(_ThreadShutdown())
        await asyncio.to_thread(self._thread.join)
