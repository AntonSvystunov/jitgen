from __future__ import annotations

import asyncio

from .base import BaseExecutor, SourceCode, StatementExtractor


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

        session.reset()   # keep executor REPL state; clear buffers for next turn
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
        self._tail: asyncio.Task[None] | None = None
        self._generation: int = 0

    # ── public properties ──────────────────────────────────────────────

    @property
    def error(self) -> Exception | None:
        """First error encountered (syntax or runtime), or ``None``.

        Set synchronously on syntax errors inside :meth:`push`, and
        asynchronously by the background execution task on runtime errors.
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
        schedule their execution as a background task (fire-and-forget).

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
            self._schedule(stmt)

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
                    self._schedule(stmt)

        if self._tail is not None:
            await self._tail

        if self._error is not None:
            raise self._error

        return "".join(self._output_parts)

    def reset(self) -> None:
        """Clear buffer, aggregated output, captured error, and the pending
        task chain.  The executor retains its own state (REPL ``_locals``).

        Call at the start of each agent turn to prepare for the next stream.
        """
        self._generation += 1
        if self._tail is not None and not self._tail.done():
            self._tail.cancel()
        self._buffer = ""
        self._output_parts = []
        self._error = None
        self._tail = None

    # ── private helpers ────────────────────────────────────────────────

    def _schedule(self, statement: SourceCode) -> None:
        """Chain *statement* execution onto the existing tail task."""
        prev = self._tail
        gen = self._generation

        async def _run() -> None:
            if prev is not None:
                try:
                    await prev
                except Exception:
                    pass  # error already captured in self._error
            if self._generation != gen or self._error is not None:
                return
            try:
                exec_result = await self._executor.aexecute(statement)
            except Exception as exc:
                if self._generation == gen:
                    self._error = exc
                return
            if self._generation != gen:
                return
            if not exec_result.success:
                self._error = RuntimeError(exec_result.error or "Execution failed")
                return
            if exec_result.output:
                self._output_parts.append(exec_result.output)

        self._tail = asyncio.create_task(_run())
