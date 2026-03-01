from lark import (
    Lark,
    Token,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedToken,
    UnexpectedInput,
)
from lark.tree import Branch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..base import BaseExecutor


class JITGenV2(BaseModel):
    """Push-based JIT code generation and execution engine.

    Unlike :class:`JITGen` which consumes an ``Iterator``/``AsyncIterator``,
    ``JITGenV2`` is stateful: callers push code chunks one at a time via
    :meth:`push` (sync) or :meth:`apush` (async). Each call returns any
    stdout output produced by newly-executed statements.

    After all chunks have been pushed, call :meth:`flush` / :meth:`aflush`
    to execute whatever remains in the parse buffer.
    """

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

    # ── private state ──────────────────────────────────────────────────
    _interpreter: BaseExecutor = PrivateAttr()
    _code_buffer: str = PrivateAttr(default="")
    _chunks: list[str] = PrivateAttr(default_factory=list)
    _has_executed: bool = PrivateAttr(default=False)
    _has_output: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: object) -> None:
        self._interpreter = self.interpreter_type()

    # ── public read-only properties ────────────────────────────────────

    @property
    def chunks(self) -> list[str]:
        """All chunks that have been pushed so far (defensive copy)."""
        return list(self._chunks)

    @property
    def has_executed(self) -> bool:
        """``True`` if at least one statement has been executed."""
        return self._has_executed

    @property
    def has_output(self) -> bool:
        """``True`` if at least one execution produced non-empty output."""
        return self._has_output

    @property
    def code_buffer(self) -> str:
        """The current unparsed buffer (useful for debugging)."""
        return self._code_buffer

    # ── push / apush ───────────────────────────────────────────────────

    def push(self, chunk: str, *, timeout: float = 5.0) -> str:
        """Push a code *chunk* and synchronously execute any complete statements.

        Returns concatenated stdout from all statements that were executed
        (empty string when nothing ran).
        """
        self._chunks.append(chunk)
        self._code_buffer += chunk
        return self._process_buffer(timeout=timeout, flush=False)

    async def apush(self, chunk: str, *, timeout: float = 5.0) -> str:
        """Async counterpart of :meth:`push`."""
        self._chunks.append(chunk)
        self._code_buffer += chunk
        return await self._aprocess_buffer(timeout=timeout, flush=False)

    # ── flush / aflush ─────────────────────────────────────────────────

    def flush(self, *, timeout: float = 5.0) -> str:
        """Parse and execute whatever remains in the buffer.

        Call this once after all chunks have been pushed.  Returns
        concatenated stdout from all executed statements.
        """
        return self._process_buffer(timeout=timeout, flush=True)

    async def aflush(self, *, timeout: float = 5.0) -> str:
        """Async counterpart of :meth:`flush`."""
        return await self._aprocess_buffer(timeout=timeout, flush=True)

    # ── private helpers ────────────────────────────────────────────────

    def _execute_statement(
        self,
        buffer: str,
        statement: Branch[Token],
        timeout: float = 5.0,
    ) -> str:
        """Execute a single statement and update tracking flags."""
        meta = getattr(statement, "meta", None)  # type: ignore[arg-type]
        if meta is None:
            return ""
        start = getattr(meta, "start_pos", 0)  # type: ignore[arg-type]
        end = getattr(meta, "end_pos", 0)  # type: ignore[arg-type]
        stmt = buffer[start:end]

        if stmt.strip():
            self._has_executed = True
            execution_result = self._interpreter.execute(stmt, timeout=timeout)
            if not execution_result.success:
                raise ValueError(
                    f"Error detected. Halting further processing. {execution_result.error}"
                )
            output = execution_result.output or ""
            if output:
                self._has_output = True
            return output
        return ""

    async def _aexecute_statement(
        self,
        buffer: str,
        statement: Branch[Token],
        timeout: float = 5.0,
    ) -> str:
        """Async counterpart of :meth:`_execute_statement`."""
        meta = getattr(statement, "meta", None)  # type: ignore[arg-type]
        if meta is None:
            return ""
        start = getattr(meta, "start_pos", 0)  # type: ignore[arg-type]
        end = getattr(meta, "end_pos", 0)  # type: ignore[arg-type]
        stmt = buffer[start:end]

        if stmt.strip():
            self._has_executed = True
            execution_result = await self._interpreter.aexecute(stmt, timeout=timeout)
            if not execution_result.success:
                raise ValueError(
                    f"Error detected. Halting further processing. {execution_result.error}"
                )
            output = execution_result.output or ""
            if output:
                self._has_output = True
            return output
        return ""

    def _process_buffer(self, *, timeout: float, flush: bool) -> str:
        """Shared sync logic for :meth:`push` and :meth:`flush`.

        When *flush* is ``False`` (push mode), only confirmed-complete
        statements are executed (all children except the last).  When
        *flush* is ``True``, *all* remaining children are executed and the
        buffer is cleared.
        """
        if not self._code_buffer.strip():
            if flush:
                self._code_buffer = ""
            return ""

        try:
            tree = self.parser.parse(self._code_buffer)
        except UnexpectedEOF:
            if flush:
                raise ValueError(
                    "Syntax error detected. Halting further processing. "
                    "Unexpected end of input during flush."
                )
            return ""
        except UnexpectedCharacters as e:
            raise ValueError(
                f"Syntax error detected. Halting further processing. {str(e)}"
            )
        except UnexpectedToken as e:
            token_type = getattr(e.token, "type", None)
            if not flush and token_type in self.indentation_tokens:
                return ""
            raise ValueError(
                f"Syntax error detected. Halting further processing. {str(e)}"
            )

        if flush:
            # Execute every child and clear the buffer.
            output_parts: list[str] = []
            for statement in tree.children:
                output_parts.append(
                    self._execute_statement(self._code_buffer, statement, timeout)
                )
            self._code_buffer = ""
            return "".join(output_parts)

        # Push mode: need at least 2 children to consider the first N-1 complete.
        if len(tree.children) < 2:
            return ""

        output_parts = []
        executed_upto = 0
        for statement in tree.children[:-1]:
            output_parts.append(
                self._execute_statement(self._code_buffer, statement, timeout)
            )
            meta = getattr(statement, "meta", None)  # type: ignore[arg-type]
            if meta is not None:
                executed_upto = getattr(meta, "end_pos", 0)  # type: ignore[arg-type]
        self._code_buffer = self._code_buffer[executed_upto:]
        return "".join(output_parts)

    async def _aprocess_buffer(self, *, timeout: float, flush: bool) -> str:
        """Async counterpart of :meth:`_process_buffer`."""
        if not self._code_buffer.strip():
            if flush:
                self._code_buffer = ""
            return ""

        try:
            tree = self.parser.parse(self._code_buffer)
        except UnexpectedEOF:
            if flush:
                raise ValueError(
                    "Syntax error detected. Halting further processing. "
                    "Unexpected end of input during flush."
                )
            return ""
        except UnexpectedCharacters as e:
            raise ValueError(
                f"Syntax error detected. Halting further processing. {str(e)}"
            )
        except UnexpectedToken as e:
            token_type = getattr(e.token, "type", None)
            if not flush and token_type in self.indentation_tokens:
                return ""
            raise ValueError(
                f"Syntax error detected. Halting further processing. {str(e)}"
            )

        if flush:
            output_parts: list[str] = []
            for statement in tree.children:
                output_parts.append(
                    await self._aexecute_statement(
                        self._code_buffer, statement, timeout
                    )
                )
            self._code_buffer = ""
            return "".join(output_parts)

        if len(tree.children) < 2:
            return ""

        output_parts = []
        executed_upto = 0
        for statement in tree.children[:-1]:
            output_parts.append(
                await self._aexecute_statement(
                    self._code_buffer, statement, timeout
                )
            )
            meta = getattr(statement, "meta", None)  # type: ignore[arg-type]
            if meta is not None:
                executed_upto = getattr(meta, "end_pos", 0)  # type: ignore[arg-type]
        self._code_buffer = self._code_buffer[executed_upto:]
        return "".join(output_parts)
