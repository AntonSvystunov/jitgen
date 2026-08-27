import asyncio
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from .base import BaseExecutor, SourceCode, StatementExtractor
from .errors import ExecutionError, ExtractionError, JitGenError


@dataclass(slots=True)
class _WorkItem:
    statement: SourceCode
    generation: int


@dataclass(slots=True)
class SessionStats:
    """Execution telemetry, readable without wrapping the executor.

    Timestamps are absolute `time.perf_counter` readings: a session has no
    idea when the caller's stream began, and rebasing on every `Session.reset`
    would make a multi-turn agent's numbers unreadable. Use `since` to convert.

    Not cleared by `Session.reset` — a multi-turn agent wants running totals
    across turns. Call `reset` explicitly to zero them.

    Attributes:
        statements_executed: Count of statements dispatched to the executor,
            including ones that failed.
        statements_failed: Count of dispatched statements that failed.
        first_statement_at: `time.perf_counter` reading when the first
            statement finished executing, or `None` if none has yet.
        last_statement_at: `time.perf_counter` reading when the most recent
            statement finished executing, or `None` if none has yet.
        execution_seconds: Total wall-clock time spent inside the executor,
            summed across all dispatched statements.
        early_exit: Whether the stream ended before its natural completion.
    """

    statements_executed: int = 0
    statements_failed: int = 0
    first_statement_at: float | None = None
    last_statement_at: float | None = None
    execution_seconds: float = 0.0
    early_exit: bool = False

    def since(self, start: float, default: float | None = None) -> float | None:
        """Compute seconds from `start` until the first statement finished.

        Args:
            start: A `time.perf_counter` reading to measure from.
            default: Value to return when nothing ever ran.

        Returns:
            `first_statement_at - start`, or `default` when nothing ever
            ran — no code block in the stream, or the stream died first.
        """
        if self.first_statement_at is None:
            return default
        return self.first_statement_at - start

    def reset(self) -> None:
        """Zero every counter and timestamp."""
        self.statements_executed = 0
        self.statements_failed = 0
        self.first_statement_at = None
        self.last_statement_at = None
        self.execution_seconds = 0.0
        self.early_exit = False


def _describe(exc: BaseException) -> str:
    """Render `exc` as `"TypeError: 'int' object is not iterable"`.

    Args:
        exc: The exception to describe.

    Returns:
        `exc`'s type name, followed by `": {message}"` when it has one.
    """
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _may_close_statement(buffer: SourceCode, appended_at: int) -> bool:
    """Check whether the text appended at `appended_at` could have closed a statement.

    A cheap, grammar-independent necessary condition for
    `StatementExtractor.extract` to return anything new. Only two events can
    move a statement boundary:

    * a newline arrives — the statement it terminates may now be complete; or
    * a non-blank character lands in column 0 — a *new* top-level statement
      starts, which is what lets an extractor confirm the preceding one.

    Scans only the appended span, so cost is proportional to the chunk rather
    than the buffer. Conservative in the safe direction: when it returns
    `True` the extractor still decides, and it only returns `False` when
    neither event occurred.

    Args:
        buffer: The full buffer after the append.
        appended_at: Index in `buffer` where the newly appended text starts.

    Returns:
        `False` only when neither boundary-moving event occurred in the
        appended span, meaning the extractor cannot possibly have new work.
    """
    for i in range(appended_at, len(buffer)):
        char = buffer[i]
        if char == "\n":
            return True
        if char not in " \t\r" and (i == 0 or buffer[i - 1] == "\n"):
            return True
    return False


class Session:
    """Grammar-agnostic, executor-agnostic JITGen session.

    Feed it source code as it streams; it extracts complete top-level statements
    and dispatches them to the executor in order, on a background worker.
    `push` is synchronous and returns immediately, so reading the model stream
    is never blocked by execution.

    For the common case of driving this from a raw LLM stream, use
    `StreamDriver` (`jitgen.driver.StreamDriver`), which pairs a session with a
    `CodeSegmenter` (`jitgen.base.CodeSegmenter`) and owns the loop:

    ```python
    async with create_python_session() as session:
        driver = StreamDriver(session, markdown_code("python"))
        async for chunk in llm_stream:
            for output in await driver.apush(chunk):
                ...
            if driver.has_error:
                break              # abort inference — the point of all this
        for output in await driver.afinish():
            ...
    ```

    Args:
        extractor: Grammar-aware converter from buffered source to statements.
        executor: Runs each dispatched statement, persisting state between them.
        owns_executor: Whether `aclose` should also close `executor`.
    """

    def __init__(
        self,
        *,
        extractor: StatementExtractor,
        executor: BaseExecutor,
        owns_executor: bool = False,
    ) -> None:
        self._extractor = extractor
        self._executor = executor
        self._owns_executor = owns_executor
        self._buffer: SourceCode = ""
        self._output_parts: list[str] = []
        self._settled: str | None = None
        self._error: JitGenError | None = None
        self._generation: int = 0
        self._queue: asyncio.Queue[_WorkItem] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self.stats = SessionStats()

    # ── public properties ──────────────────────────────────────────────

    @property
    def error(self) -> JitGenError | None:
        """First error encountered, or `None`.

        Set synchronously by `push` for extraction failures, and
        asynchronously by the background worker for execution failures. Poll
        `has_error` after each push to abort LLM inference early.
        """
        return self._error

    @property
    def has_error(self) -> bool:
        """`True` once an extraction or execution failure has been recorded."""
        return self._error is not None

    @property
    def buffer(self) -> SourceCode:
        """The unexecuted suffix of everything pushed so far."""
        return self._buffer

    @property
    def executor(self) -> BaseExecutor:
        """The executor running dispatched statements."""
        return self._executor

    # ── public API ─────────────────────────────────────────────────────

    def push(self, source: SourceCode) -> None:
        """Append `source`, extract ready statements, schedule them, return.

        Never blocks and never raises: an extraction failure is captured into
        `error` so the caller can poll `has_error` and stop the model rather
        than handling an exception mid-stream.

        Args:
            source: The next increment of source code to append to the buffer.
        """
        if self._error is not None:
            return
        self._settled = None
        appended_at = len(self._buffer)
        self._buffer += source
        # Parsing is the only expensive part of push(), and it runs on the event
        # loop — every wasted parse is latency stolen from reading the LLM
        # stream.  Most chunks land mid-line and cannot move a statement
        # boundary, so skip them outright.
        if not _may_close_statement(self._buffer, appended_at):
            return
        self._extract(final=False)

    def take_output(self) -> str:
        """Pop stdout produced so far, without awaiting pending work.

        Lets a caller forward output the moment a statement produces it instead
        of holding everything back until `result`. Output is returned once —
        a later `result` sees only what arrives after this.

        Returns:
            The stdout accumulated since the last call to `take_output` or
            `result`; empty when there is none.
        """
        if not self._output_parts:
            return ""
        parts, self._output_parts = self._output_parts, []
        return "".join(parts)

    async def result(self) -> str:
        """Flush the buffer, await pending execution, return stdout.

        Idempotent: the outcome is computed once and cached, so calling twice
        returns the same string or re-raises the same error instead of running
        the final extraction again. The next `push` invalidates the cache.

        Returns:
            All stdout produced since the last `take_output` or `result` call.

        Raises:
            JitGenError: the first extraction or execution failure.
        """
        if self._settled is None:
            if self._error is None:
                self._extract(final=True)
            # Awaited even on the error path: leaving a statement in flight would
            # let it mutate executor state after the caller has moved on, and a
            # remote executor would still be busy when the next turn arrives.
            await self._await_quiescence()
            self._settled = "".join(self._output_parts)
            self._output_parts.clear()
        if self._error is not None:
            raise self._error
        return self._settled

    async def reset(self) -> None:
        """Clear buffer, output and error; drop queued work; keep REPL state.

        Awaits executor cancellation and worker quiescence before returning, so
        no in-flight execution can mutate shared state after this resolves.
        Call at the end of each agent turn to prepare for the next stream.
        `stats` is deliberately preserved.
        """
        self._generation += 1
        self._drain_queue()
        await self._acancel()
        try:
            await self._await_quiescence()
        finally:
            # Also runs when the caller is cancelled mid-reset, so the next turn
            # never inherits a poisoned buffer or a stale error.
            self._buffer = ""
            self._output_parts.clear()
            self._settled = None
            self._error = None

    async def aclose(self) -> None:
        """Stop the background worker, and the executor if this session owns it.

        The session stays usable afterwards: a later `push` starts a fresh
        worker. Cancellation-safe — the worker reference is dropped before
        awaiting, so an interrupted `aclose()` cannot leave a doomed worker
        behind for the next turn to adopt.
        """
        worker, self._worker = self._worker, None
        if worker is not None and not worker.done():
            worker.cancel()
            # return_exceptions=True absorbs the worker's own CancelledError
            # while still letting cancellation of *this* task propagate.
            await asyncio.gather(worker, return_exceptions=True)
        # The old queue may hold items nobody will ever consume; a later join()
        # on it would wait forever.
        self._queue = asyncio.Queue()
        if self._owns_executor:
            await self._executor.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ── extraction ─────────────────────────────────────────────────────

    def _extract(self, *, final: bool) -> None:
        """Pull ready statements out of the buffer and queue them for execution.

        Args:
            final: `True` when the stream has ended, so the last withheld
                statement should be flushed too.
        """
        try:
            statements, self._buffer = self._extractor.extract(
                self._buffer, final=final
            )
        except SyntaxError as exc:
            self._error = ExtractionError(str(exc), source=self._buffer)
            return
        for stmt in statements:
            self._queue.put_nowait(_WorkItem(stmt, self._generation))
        if statements:
            self._ensure_worker()

    # ── worker ─────────────────────────────────────────────────────────

    def _ensure_worker(self) -> None:
        """Start the background worker task if none is currently running."""
        if self._worker is None or self._worker.done():
            # The queue is passed explicitly: a worker must never outlive the
            # queue whose counter it decrements, or a task_done() would land on
            # a replacement queue and corrupt its accounting.
            self._worker = asyncio.create_task(self._worker_loop(self._queue))

    async def _worker_loop(self, queue: asyncio.Queue[_WorkItem]) -> None:
        """Consume `queue` and execute each item in order until cancelled.

        Args:
            queue: The work queue to drain. Passed explicitly rather than
                read from `self._queue` — see `_ensure_worker`.
        """
        while True:
            item = await queue.get()
            try:
                if item.generation != self._generation or self._error is not None:
                    continue
                await self._execute(item)
            finally:
                # In a finally so stale-generation skips decrement too, and so a
                # worker cancelled inside aexecute() still balances the counter
                # instead of wedging a later join().
                queue.task_done()

    async def _execute(self, item: _WorkItem) -> None:
        """Run one dispatched statement and record its outcome.

        Args:
            item: The statement to execute, tagged with the generation it
                was queued under.
        """
        started = time.perf_counter()
        error: JitGenError | None = None
        result = None
        try:
            result = await self._executor.aexecute(item.statement)
        except Exception as exc:  # noqa: BLE001
            error = ExecutionError(_describe(exc), statement=item.statement)
            error.__cause__ = exc
        finally:
            # A statement that raised still ran; excluding it would flatter the
            # time-to-first-statement of exactly the runs that failed fastest.
            self._record(started)
        if result is not None and not result.success:
            error = ExecutionError.from_result(result, statement=item.statement)
        if item.generation != self._generation:
            return  # a reset overtook us; this turn's state is no longer ours
        if error is not None:
            self._error = error
            self.stats.statements_failed += 1
        elif result is not None and result.output:
            self._output_parts.append(result.output)

    def _record(self, started: float) -> None:
        """Update `stats` for a statement that started at `started`.

        Args:
            started: `time.perf_counter` reading taken before execution began.
        """
        now = time.perf_counter()
        self.stats.statements_executed += 1
        self.stats.execution_seconds += now - started
        if self.stats.first_statement_at is None:
            self.stats.first_statement_at = now
        self.stats.last_statement_at = now

    # ── quiescence ─────────────────────────────────────────────────────

    async def _await_quiescence(self) -> None:
        """Block until everything queued so far has been consumed.

        Races `asyncio.Queue.join` against the worker task rather than
        awaiting the join alone: a worker that died leaves `unfinished_tasks`
        above zero with nobody left to decrement it, and `join()` would then
        never return. When that happens the queue is replaced outright.
        """
        if self._queue.empty() and (self._worker is None or self._worker.done()):
            return
        # Revive a dead or absent worker so queued items still run.
        self._ensure_worker()
        worker = self._worker
        assert worker is not None
        joiner = asyncio.ensure_future(self._queue.join())
        try:
            await asyncio.wait({joiner, worker}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # Give the executor a chance to interrupt in-flight work before the
            # next call arrives, without blocking this already-cancelled task.
            self._fire_acancel()
            raise
        finally:
            joiner.cancel()
            if worker.done():
                if not worker.cancelled():
                    # Consume it, or asyncio logs "exception was never retrieved".
                    worker.exception()
                self._queue = asyncio.Queue()
                self._worker = None

    def _drain_queue(self) -> None:
        """Discard queued work, keeping the unfinished counter balanced."""
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._queue.task_done()

    async def _acancel(self) -> None:
        """Best-effort cancellation of in-flight execution."""
        try:
            await self._executor.acancel()
        except Exception:  # noqa: BLE001, S110
            pass  # an executor that cannot cancel must not break reset()

    def _fire_acancel(self) -> None:
        """Schedule `_acancel` without awaiting it, for use from a cancelled task."""
        task = asyncio.get_running_loop().create_task(self._acancel())
        # _acancel never raises, but a cancelled task still needs its outcome
        # retrieved to stay out of the event loop's exception handler.
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
