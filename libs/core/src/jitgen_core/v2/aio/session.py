from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from lark import Lark
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..base import BaseExecutor
from ..base import MarkerStatefulAlgorithm

StdoutHandler = Callable[[str], Any]
ErrorHandler = Callable[[Exception], Any]
StatementHandler = Callable[[str, str], Any]


class AsyncJITGenSession(BaseModel):
    """Stateful async session that executes code between start/end markers."""

    model_config = ConfigDict(  # type: ignore[assignment]
        arbitrary_types_allowed=True,
    )

    parser: Lark = Field(
        description="The Lark parser instance used for parsing the input code."
    )

    interpreter_type: type[BaseExecutor] = Field(
        description="The interpreter class that executes parsed statements."
    )

    indentation_tokens: set[str] = Field(
        default_factory=set,
        description="Set of tokens that signal incomplete indentation in the parser.",
    )

    start_marker: str = Field(
        default="```",
        description="Marker indicating when executable content starts.",
    )

    end_marker: str = Field(
        default="```",
        description="Marker indicating when executable content ends.",
    )

    _interpreter: BaseExecutor = PrivateAttr()
    _algorithm: MarkerStatefulAlgorithm = PrivateAttr()
    _chunks: list[str] = PrivateAttr(default_factory=list)
    _has_executed: bool = PrivateAttr(default=False)
    _has_output: bool = PrivateAttr(default=False)
    _stdout_handlers: list[StdoutHandler] = PrivateAttr(default_factory=list)
    _error_handlers: list[ErrorHandler] = PrivateAttr(default_factory=list)
    _statement_handlers: list[StatementHandler] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        self._interpreter = self.interpreter_type()
        self._algorithm = MarkerStatefulAlgorithm(
            parser=self.parser,
            indentation_tokens=self.indentation_tokens,
            start_marker=self.start_marker,
            end_marker=self.end_marker,
        )

    @property
    def chunks(self) -> list[str]:
        return list(self._chunks)

    @property
    def has_executed(self) -> bool:
        return self._has_executed

    @property
    def has_output(self) -> bool:
        return self._has_output

    @property
    def code_buffer(self) -> str:
        return self._algorithm.code_buffer

    def on_stdout(self, handler: StdoutHandler) -> StdoutHandler:
        self._stdout_handlers.append(handler)
        return handler

    def on_error(self, handler: ErrorHandler) -> ErrorHandler:
        self._error_handlers.append(handler)
        return handler

    def on_statement_complete(self, handler: StatementHandler) -> StatementHandler:
        self._statement_handlers.append(handler)
        return handler

    async def apush(self, chunk: str, *, timeout: float = 5.0) -> str:
        self._chunks.append(chunk)
        output_parts: list[str] = []

        for segment in self._algorithm.ingest_chunk(chunk):
            if segment.text:
                self._algorithm.append_code(segment.text)
                output_parts.extend(
                    await self._aexecute_ready_statements(timeout=timeout, flush=False)
                )
            if segment.flush_after:
                output_parts.extend(
                    await self._aexecute_ready_statements(timeout=timeout, flush=True)
                )

        return "".join(output_parts)

    async def aflush(self, *, timeout: float = 5.0) -> str:
        output_parts: list[str] = []

        for segment in self._algorithm.finalize_ingest():
            if segment.text:
                self._algorithm.append_code(segment.text)
            if segment.flush_after:
                output_parts.extend(
                    await self._aexecute_ready_statements(timeout=timeout, flush=True)
                )

        output_parts.extend(
            await self._aexecute_ready_statements(timeout=timeout, flush=True)
        )
        return "".join(output_parts)

    async def __aenter__(self) -> AsyncJITGenSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        if exc_type is None:
            await self.aflush()
        return False

    async def _aexecute_ready_statements(
        self, *, timeout: float, flush: bool
    ) -> list[str]:
        statements = self._algorithm.pop_ready_statements(flush=flush)
        outputs: list[str] = []
        for statement in statements:
            execution_result = await self._interpreter.aexecute(statement, timeout=timeout)
            if not execution_result.success:
                error = ValueError(
                    f"Error detected. Halting further processing. {execution_result.error}"
                )
                await self._aemit_error(error)
                raise error

            self._has_executed = True
            output = execution_result.output or ""
            if output:
                self._has_output = True
                await self._aemit_stdout(output)
            await self._aemit_statement_complete(statement, output)
            outputs.append(output)
        return outputs

    async def _arun_handler(self, callback_result: Any) -> None:
        if inspect.isawaitable(callback_result):
            await callback_result

    async def _aemit_stdout(self, output: str) -> None:
        for handler in self._stdout_handlers:
            await self._arun_handler(handler(output))

    async def _aemit_error(self, error: Exception) -> None:
        for handler in self._error_handlers:
            await self._arun_handler(handler(error))

    async def _aemit_statement_complete(self, statement: str, output: str) -> None:
        for handler in self._statement_handlers:
            await self._arun_handler(handler(statement, output))
