from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from lark import Lark
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..base import BaseExecutor
from .base import MarkerStatefulAlgorithm

StdoutHandler = Callable[[str], Any]
ErrorHandler = Callable[[Exception], Any]
StatementHandler = Callable[[str, str], Any]
T = TypeVar("T")


class JITGenSession(BaseModel):
    """Stateful sync session that executes code between start/end markers."""

    model_config = ConfigDict(  # type: ignore[assignment]
        arbitrary_types_allowed=True,
    )

    parser: Lark = Field(
        description="The Lark parser instance used for parsing the input code."
    )

    interpreter_type: type[BaseExecutor] = Field(
        description="The interpreter class that executes parsed statements."
    )

    interpreter_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed when instantiating the interpreter.",
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
        self._interpreter = self.interpreter_type(**self.interpreter_kwargs)
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

    def push(self, chunk: str, *, timeout: float = 5.0) -> str:
        self._chunks.append(chunk)
        output_parts: list[str] = []

        for segment in self._run_algorithm_call(lambda: self._algorithm.ingest_chunk(chunk)):
            if segment.text:
                self._run_algorithm_call(lambda: self._algorithm.append_code(segment.text))
                output_parts.extend(self._execute_ready_statements(timeout=timeout, flush=False))
            if segment.flush_after:
                output_parts.extend(self._execute_ready_statements(timeout=timeout, flush=True))

        return "".join(output_parts)

    def flush(self, *, timeout: float = 5.0) -> str:
        output_parts: list[str] = []

        for segment in self._run_algorithm_call(self._algorithm.finalize_ingest):
            if segment.text:
                self._run_algorithm_call(lambda: self._algorithm.append_code(segment.text))
            if segment.flush_after:
                output_parts.extend(self._execute_ready_statements(timeout=timeout, flush=True))

        output_parts.extend(self._execute_ready_statements(timeout=timeout, flush=True))
        return "".join(output_parts)

    def __enter__(self) -> JITGenSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        if exc_type is None:
            self.flush()
        return False

    def _execute_ready_statements(self, *, timeout: float, flush: bool) -> list[str]:
        statements = self._run_algorithm_call(
            lambda: self._algorithm.pop_ready_statements(flush=flush)
        )
        outputs: list[str] = []
        for statement in statements:
            execution_result = self._interpreter.execute(statement, timeout=timeout)
            if not execution_result.success:
                error = ValueError(
                    f"Error detected. Halting further processing. {execution_result.error}"
                )
                self._emit_error(error)
                raise error

            self._has_executed = True
            output = execution_result.output or ""
            if output:
                self._has_output = True
                self._emit_stdout(output)
            self._emit_statement_complete(statement, output)
            outputs.append(output)
        return outputs

    def _run_algorithm_call(self, operation: Callable[[], T]) -> T:
        try:
            return operation()
        except Exception as error:
            self._emit_error(error)
            raise

    def _run_handler_sync(self, callback_result: Any) -> None:
        if not inspect.isawaitable(callback_result):
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(callback_result)
            return

        raise RuntimeError(
            "Async handlers in sync sessions require no running event loop. "
            "Use jitgen_core.aio.AsyncJITGenSession inside async applications."
        )

    def _emit_stdout(self, output: str) -> None:
        for handler in self._stdout_handlers:
            self._run_handler_sync(handler(output))

    def _emit_error(self, error: Exception) -> None:
        for handler in self._error_handlers:
            self._run_handler_sync(handler(error))

    def _emit_statement_complete(self, statement: str, output: str) -> None:
        for handler in self._statement_handlers:
            self._run_handler_sync(handler(statement, output))
