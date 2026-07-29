from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .base import BaseExecutor, SourceCode, StatementExtractor


@dataclass
class _WorkItem:
    statement: SourceCode
    generation: int


@dataclass
class _Sentinel:
    future: asyncio.Future[None]


class Session:
    """Grammar-agnostic, executor-agnostic JITGen session.

    Usage pattern for ReAct/CodeAct agent loops::

        session = create_python_jitgen(executor=my_executor)
        stripper = MarkerStripper(start='{"code":"', end='"}')

        # During LLM streaming:
        async for chunk in llm_stream:
            for seg in stripper.process(chunk):
                session.push(seg.text)   # fire-and-forget
            if session.has_error:
                break                    # abort inference early

        # At tool-call boundary:
        try:
            output = await session.result()
        except Exception as e:
            output = str(e)

        await session.reset()   # keep executor REPL state; clear buffers for next turn
        stripper.reset()
    """

    def __init__(
        self, *, extractor: StatementExtractor, executor: BaseExecutor
    ) -> None:
        self._extractor = extractor
        self._executor = executor
        self._buffer: SourceCode = ""
        self._output_parts: list[str] = []
        self._error: Exception | None = None
        self._generation: int = 0
        self._queue: asyncio.Queue[_WorkItem | _Sentinel] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    # ── public properties ──────────────────────────────────────────────

    @property
    def error(self) -> Exception | None:
        """First error encountered (syntax or runtime), or ``None``.

        Set synchronously on syntax errors inside :meth:`push`, and
        asynchronously by the background worker on runtime errors.
        Poll this after each :meth:`push` to abort LLM inference early.
        """
        return self._error

    @property
    def has_error(self) -> bool:
        return self._error is not None

    @property
    def buffer(self) -> SourceCode:
        return self._buffer

    # ── public API ─────────────────────────────────────────────────────

    def push(self, source: SourceCode) -> None:
        """Append *source* to the buffer, extract ready statements, and
        schedule their execution via the background worker (fire-and-forget).

        Returns immediately without awaiting execution.  Syntax errors from
        the extractor are captured into :attr:`error` (never raised) so that
        callers can poll :attr:`has_error` to abort an LLM stream early.
        """
        if self._error is not None:
            return
        self._buffer += source
        try:
            statements, self._buffer = self._extractor.extract(
                self._buffer, final=False
            )
        except SyntaxError as exc:
            self._error = exc
            return
        for stmt in statements:
            self._queue.put_nowait(_WorkItem(stmt, self._generation))
        if statements:
            self._ensure_worker()

    async def result(self) -> str:
        """Flush remaining buffer, await all pending executions, and return
        the concatenated stdout produced since the last :meth:`reset`.

        Raises the first captured error (syntax or runtime).
        """
        if self._error is None:
            try:
                statements, self._buffer = self._extractor.extract(
                    self._buffer, final=True
                )
            except SyntaxError as exc:
                self._error = exc
            else:
                for stmt in statements:
                    self._queue.put_nowait(_WorkItem(stmt, self._generation))
                if statements:
                    self._ensure_worker()

        if not self._queue.empty() or (
            self._worker is not None and not self._worker.done()
        ):
            # Revive a dead/absent worker so queued items still run and the
            # sentinel is guaranteed to be consumed.
            self._ensure_worker()
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._queue.put_nowait(_Sentinel(fut))
            try:
                await fut
            except asyncio.CancelledError:
                self._fire_acancel()
                raise

        if self._error is not None:
            raise self._error

        return "".join(self._output_parts)

    async def reset(self) -> None:
        """Clear buffer, aggregated output, captured error, and drain the pending
        queue.  The executor retains its own state (REPL ``_locals``).

        Awaits executor cancellation and worker quiescence before returning,
        so no in-flight execution mutates shared state after this call resolves.

        Call at the end of each agent turn to prepare for the next stream.
        """
        self._generation += 1
        # Drain pending items so the worker skips them quickly.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Best-effort cancellation of any in-flight executor call.
        acancel = getattr(self._executor, "acancel", None)
        if acancel is not None:
            try:
                await acancel()
            except Exception:
                pass
        # Wait for the worker to finish any in-flight aexecute and reach a
        # clean pause point before we wipe state.
        if self._worker is not None and not self._worker.done():
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._queue.put_nowait(_Sentinel(fut))
            try:
                await fut
            except asyncio.CancelledError:
                # Outer task was cancelled while waiting; fire another acancel
                # so the worker unblocks faster, clear state, then re-raise.
                self._fire_acancel()
                self._buffer = ""
                self._output_parts = []
                self._error = None
                raise
        elif self._worker is not None:
            # Worker already finished (e.g. cancelled mid-turn); drop the stale
            # reference so the next push() starts a fresh one.
            self._worker = None
        # Clear session state; executor state (REPL _locals) is preserved.
        self._buffer = ""
        self._output_parts = []
        self._error = None

    async def aclose(self) -> None:
        """Stop the background worker task.

        Call when the session is no longer needed to avoid pending-task warnings
        on event-loop close.  The session remains usable afterwards: a later
        :meth:`push` / :meth:`result` transparently starts a fresh worker.

        Cancellation-safe — the worker reference is dropped before awaiting, so
        an interrupted ``aclose()`` cannot leave a doomed worker behind for the
        next turn to adopt.
        """
        worker, self._worker = self._worker, None
        if worker is None or worker.done():
            return
        worker.cancel()
        # return_exceptions=True absorbs the worker's own CancelledError while
        # still letting cancellation of *this* task propagate.
        await asyncio.gather(worker, return_exceptions=True)

    # ── private helpers ────────────────────────────────────────────────

    def _fire_acancel(self) -> None:
        """Schedule executor.acancel() as a background task (fire-and-forget).

        Used when the caller is being cancelled so we do not block on the
        cleanup, but still give the executor a chance to interrupt in-flight
        work before the next call arrives.
        """
        acancel = getattr(self._executor, "acancel", None)
        if acancel is not None:
            asyncio.get_running_loop().create_task(acancel())

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        while True:
            item = await self._queue.get()
            if isinstance(item, _Sentinel):
                if not item.future.done():
                    item.future.set_result(None)
                continue
            # _WorkItem: skip if stale generation or a prior error was recorded.
            if item.generation != self._generation or self._error is not None:
                continue
            try:
                exec_result = await self._executor.aexecute(item.statement)
            except Exception as exc:
                if item.generation == self._generation:
                    self._error = exc
                continue
            if item.generation != self._generation:
                continue
            if not exec_result.success:
                self._error = RuntimeError(exec_result.error or "Execution failed")
                continue
            if exec_result.output:
                self._output_parts.append(exec_result.output)
