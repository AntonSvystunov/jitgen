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
    timeout_value: float | None = Field(
        default=None, description="The timeout value in seconds, if applicable"
    )

@runtime_checkable
class BaseExecutor(Protocol):
    def execute(
        self, source_code: SourceCode, *, timeout: float
    ) -> ExecutionResult: ...

    async def aexecute(
        self, source_code: SourceCode, *, timeout: float
    ) -> ExecutionResult: ...
