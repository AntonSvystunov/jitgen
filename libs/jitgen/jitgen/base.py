from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

SourceCode = str


class ExecutionResult(BaseModel):
    """Outcome of running one fragment of source code."""

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
    has_cancelled: bool = Field(
        default=False,
        description="Indicates if the execution was cancelled via acancel()",
    )


class CodeSegment(BaseModel):
    """A slice of code text extracted from the raw model stream.

    `end_of_block=True` means this segment closes a complete code block — the
    driver resolves the session and resets for whatever comes next.
    """

    text: str = Field(description="Code text belonging to the current block.")
    end_of_block: bool = Field(
        default=False, description="Whether this segment closes the block."
    )


@runtime_checkable
class BaseExecutor(Protocol):
    """Async executor contract.

    Timeout and any other runtime knobs are owned by the executor instance
    (typically configured in `__init__`); callers simply submit source code.

    All three methods are required. `acancel` and `aclose` used to be
    discovered by `getattr` at the call site, which meant type checkers could
    not help and every decorator had to forward them blindly. Inherit
    `ExecutorBase` (`jitgen.executors.base.ExecutorBase`) to get no-op
    implementations.
    """

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        """Run `source_code` and report what happened.

        Args:
            source_code: One dispatched top-level statement.

        Returns:
            The outcome of running `source_code`.
        """
        ...

    async def acancel(self) -> None:
        """Interrupt the statement currently executing, if any."""
        ...

    async def aclose(self) -> None:
        """Release resources held by this executor."""
        ...


@runtime_checkable
class StatementExtractor(Protocol):
    """Grammar-aware converter of an evolving source buffer into ready statements.

    Implementations own their full recovery policy for a specific grammar: which
    parse errors mean "the buffer is incomplete, give me more" and which mean
    "this can never become valid". Raise `SyntaxError` on unrecoverable input.
    """

    def extract(
        self, source: SourceCode, *, final: bool
    ) -> tuple[list[SourceCode], SourceCode]:
        """Pull whatever complete top-level statements `source` now contains.

        Args:
            source: The full unexecuted-suffix buffer accumulated so far.
            final: `True` when the stream has ended, so the caller wants
                every remaining statement flushed rather than the last one
                withheld as still-growing.

        Returns:
            A tuple of `(statements, leftover_buffer)`: statements ready to
            dispatch, in order, and the suffix of `source` still awaiting
            more input.

        Raises:
            SyntaxError: `source` cannot become valid input for the grammar.
        """
        ...


@runtime_checkable
class CodeSegmenter(Protocol):
    """Extracts code text from a raw model stream, one chunk at a time.

    This is the strategy-dependent half of the method: the same grammar and
    executor work whether the model emits a markdown fence or a streaming
    tool-call argument, but recovering the code from those two differs entirely.

    Implementations buffer whatever they cannot yet classify (a partial marker,
    a half-delivered escape sequence) and must be safe to reuse after `reset`.
    """

    def feed(self, chunk: str) -> list[CodeSegment]:
        """Append `chunk` and return any newly extractable code segments.

        Args:
            chunk: The next increment of raw model output.

        Returns:
            Code segments that became extractable as a result of `chunk`, in
            order; empty when nothing is ready yet.
        """
        ...

    def finalize(self) -> list[CodeSegment]:
        """Flush whatever is buffered at end-of-stream.

        Returns:
            Any code segments still held back, in order.
        """
        ...

    def reset(self) -> None:
        """Clear buffered state so the segmenter can be reused for a new stream."""
        ...

    @property
    def inside_block(self) -> bool:
        """`True` when the stream position is inside a marker block."""
        ...
