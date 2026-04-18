from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

SourceCode = str


class ExecutionResult(BaseModel):
    success: bool = Field(description="Indicates if the execution was successful")
    output: str | None = Field(
        default=None, description="The output (stdout) of the execution, if any"
    )
    error: str | None = Field(
        default=None,
        description="Any error message (stderr) from the execution, if applicable",
    )
    has_timed_out: bool = Field(
        default=False, description="Indicates if the execution timed out"
    )


@runtime_checkable
class BaseExecutor(Protocol):
    """Async executor contract.

    Timeout and any other runtime knobs are owned by the executor instance
    (typically configured in ``__init__``). Callers simply submit source code.
    """

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult: ...


@runtime_checkable
class StatementExtractor(Protocol):
    """Grammar-aware converter of an evolving source buffer into ready statements.

    Implementations own their full recovery policy for a specific grammar:
    what parse errors mean "the buffer is incomplete, give me more" vs.
    "this is an unrecoverable syntax error". Return
    ``(statements, leftover_buffer)``; raise :class:`SyntaxError` on
    unrecoverable input.
    """

    def extract(
        self, source: SourceCode, *, final: bool
    ) -> tuple[list[SourceCode], SourceCode]: ...
