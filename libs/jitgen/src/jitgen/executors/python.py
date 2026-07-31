import asyncio
import contextlib
import ctypes
import io
import queue
import sys
import threading
import warnings
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


class ExecutionInterrupted(BaseException):
    """Injected into the worker thread to abort runaway code.

    Derives from :class:`BaseException` so that a generated ``except Exception``
    cannot swallow it.  A bare ``except:`` still can — see
    :meth:`InProcPythonExecutor.aclose` for what happens then.
    """


class ExecutionCancelled(Exception):
    """Reported when a job was interrupted before it finished."""


def _raise_in_thread(thread_id: int, exctype: type[BaseException]) -> bool:
    """Ask CPython to raise *exctype* in the thread identified by *thread_id*.

    The interpreter delivers the exception at the next bytecode boundary, which
    breaks a pure-Python loop promptly — that is what makes a runaway
    ``while True:`` recoverable at all.  It cannot interrupt a blocking call
    inside C (``time.sleep``, socket reads, ``input()``): those run to
    completion first.

    Returns whether the exception was successfully scheduled.
    """
    changed = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(exctype)
    )
    if changed > 1:
        # Documented contract: anything above 1 means we hit more thread states
        # than intended and must undo it, or we poison unrelated threads.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
        return False
    return changed == 1


def _discard_outcome(future: asyncio.Future[None]) -> None:
    """Retrieve a future's outcome so asyncio does not log it as unhandled."""
    if not future.cancelled():
        future.exception()


def _restore_std_streams() -> None:
    """Undo a ``redirect_stdout``/``redirect_stderr`` left behind by a dead job.

    Those context managers swap the streams **process-wide**, so a job that never
    unwinds keeps every later write in the process — progress bars, logging,
    tracebacks — disappearing into its own buffer.
    """
    if sys.stdout is not sys.__stdout__:
        sys.stdout = sys.__stdout__
    if sys.stderr is not sys.__stderr__:
        sys.stderr = sys.__stderr__


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

    A timed-out or cancelled job is **interrupted**: an exception is injected
    into the worker thread so a runaway ``while True:`` stops instead of spinning
    for the life of the process.  This reaches any pure-Python loop but cannot
    break a blocking C call; see :func:`_raise_in_thread`.

    .. note::
        In-process execution still cannot *guarantee* a kill.  Code that blocks
        in C, or that swallows :class:`BaseException`, survives interruption; the
        worker is then abandoned rather than joined.  Use a subprocess-based
        executor when a hard kill is required.
    """

    timeout: float = Field(
        default=60.0,
        description="Max seconds allowed per aexecute call.",
    )
    interrupt_grace: float = Field(
        default=0.1,
        description=(
            "Max seconds a timed-out aexecute waits for its interrupt to take "
            "effect before returning, so the job releases the process-wide "
            "stdout redirect first.  Kept short deliberately: an interruptible "
            "job stops at the next bytecode, and a job blocked in C will not "
            "stop within any grace period, so waiting longer only adds latency."
        ),
    )
    close_timeout: float = Field(
        default=5.0,
        description=(
            "Max seconds aclose() waits for the worker thread to stop before "
            "abandoning it.  Bounded so an uninterruptible job cannot hang the "
            "caller."
        ),
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
    _lock: threading.Lock = PrivateAttr(default=None)
    _running: _Job | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.register_tools(self.tools)
        self._thread_queue = queue.Queue()
        self._lock = threading.Lock()
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
            try:
                job = self._thread_queue.get()
                if isinstance(job, _ThreadShutdown):
                    return
                self._run_job(job)
            except ExecutionInterrupted:
                # An interrupt that arrived just after its job had already
                # finished.  Swallow it here: left to propagate it would kill the
                # worker, and every later statement would find no thread to run
                # on.
                continue

    def _run_job(self, job: _Job) -> None:
        with self._lock:
            self._running = job
        try:
            with contextlib.redirect_stdout(job.stdout):
                with contextlib.redirect_stderr(job.stderr):
                    exec(job.compiled_code, self._locals)  # noqa: S102
        except ExecutionInterrupted:
            self._settle(job, ExecutionCancelled("Execution interrupted"))
        except SystemExit as exc:
            self._settle(job, RuntimeError(f"SystemExit: {exc.code}"))
        except BaseException as exc:  # noqa: BLE001
            self._settle(job, exc)
        else:
            self._settle(job, None)
        finally:
            with self._lock:
                self._running = None

    @staticmethod
    def _settle(job: _Job, exc: BaseException | None) -> None:
        """Resolve *job*'s future from the worker thread, at most once.

        The double-settle guard matters because an interrupt can land between
        ``exec`` returning and this call, sending the same job down two paths.
        """
        def _apply() -> None:
            if job.future.done():
                return
            if exc is None:
                job.future.set_result(None)
            else:
                job.future.set_exception(exc)

        try:
            job.loop.call_soon_threadsafe(_apply)
        except RuntimeError:
            pass  # event loop already closed; nobody is waiting

    def _interrupt(self, job: _Job | None = None) -> bool:
        """Inject :class:`ExecutionInterrupted` into the running job.

        When *job* is given the interrupt only fires if that exact job is still
        the one executing, so a call that has already timed out cannot abort an
        unrelated statement that started in the meantime.
        """
        with self._lock:
            running = self._running
            if running is None or (job is not None and running is not job):
                return False
            thread_id = self._thread.ident if self._thread is not None else None
        if thread_id is None:
            return False
        return _raise_in_thread(thread_id, ExecutionInterrupted)

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        """Execute *source_code* asynchronously, returning an :class:`~jitgen_core.ExecutionResult`.

        Submits the compiled code to the dedicated worker thread and awaits
        completion.  Jobs are serialized on that one thread, so ``_locals`` is
        never mutated by two threads concurrently.

        On timeout the job is interrupted rather than left running, so a runaway
        loop does not keep a core busy — and, just as importantly, does not keep
        holding the process-wide stdout redirect.
        """
        stdout_collector = io.StringIO()
        stderr_collector = io.StringIO()
        has_error = False
        has_timed_out = False
        has_cancelled = False
        error_message: str | None = None

        try:
            compiled_code = compile(source_code, "<string>", "exec", optimize=2)
        except Exception as exc:
            return ExecutionResult(success=False, error=str(exc))

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        job = _Job(compiled_code, stdout_collector, stderr_collector, loop, future)
        self._thread_queue.put_nowait(job)

        try:
            # shield prevents wait_for from cancelling the future when the
            # timeout fires — the worker thread keeps its reference and will
            # resolve it once the job stops.
            await asyncio.wait_for(asyncio.shield(future), self.timeout)
        except asyncio.TimeoutError:
            has_timed_out = True
            has_error = True
            # str(asyncio.TimeoutError()) is empty, so callers that only forward
            # the message would report a blank failure and lose the fact that it
            # was a timeout at all.
            error_message = f"Execution timed out after {self.timeout}s"
            self._interrupt(job)
            # Give the interrupt a moment to actually land.  Until the job
            # unwinds, it still owns the process-wide stdout/stderr redirect, so
            # returning immediately would silently swallow whatever the caller
            # logs about the timeout.  An interruptible job settles in
            # microseconds; an uninterruptible one costs this grace period once.
            await asyncio.wait([future], timeout=self.interrupt_grace)
            # We stopped awaiting this future, but the worker will still settle
            # it; claim the outcome so asyncio does not report it as an
            # exception that was never retrieved.
            future.add_done_callback(_discard_outcome)
        except ExecutionCancelled:
            has_cancelled = True
            has_error = True
            error_message = "Execution cancelled"
        except Exception as exc:
            has_error = True
            error_message = str(exc) or type(exc).__name__

        return ExecutionResult(
            success=not has_error,
            output=stdout_collector.getvalue() or None,
            error=error_message,
            has_timed_out=has_timed_out,
            has_cancelled=has_cancelled,
        )

    async def acancel(self) -> None:
        """Interrupt the statement currently executing, if any.

        Returns once the interrupt has been *scheduled*; the worker stops at the
        next bytecode boundary.  Uninterruptible jobs (blocking C calls) keep
        running, and the single-worker serialization still guarantees that the
        next ``aexecute`` cannot start until they finish, so ``_locals`` stays
        consistent either way.
        """
        self._interrupt()

    async def aclose(self) -> None:
        """Stop the worker thread, without ever blocking indefinitely.

        Interrupts any in-flight job so it can reach the shutdown sentinel, then
        joins for at most :attr:`close_timeout`.  A job that refuses to die —
        blocked in C, or swallowing :class:`BaseException` — leaves the thread
        abandoned instead of hanging the caller; it is a daemon, so it will not
        keep the process alive.  In that case the process-wide stdout/stderr
        redirect it still owns is force-restored, otherwise every subsequent
        write in this process would vanish into the dead job's buffer.
        """
        self._thread_queue.put_nowait(_ThreadShutdown())
        self._interrupt()
        await asyncio.to_thread(self._thread.join, self.close_timeout)
        if self._thread.is_alive():
            _restore_std_streams()
            warnings.warn(
                "InProcPythonExecutor: worker thread did not stop within "
                f"{self.close_timeout}s and was abandoned — executed code is "
                "blocking somewhere the interpreter cannot interrupt. "
                "Use a subprocess-based executor if a hard kill is required.",
                RuntimeWarning,
                stacklevel=2,
            )
