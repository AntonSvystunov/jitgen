from typing import Self

from .base import ExecutionResult, SourceCode


class JitGenError(Exception):
    """Base class for every failure surfaced by a session.

    The timeout/cancel flags live on the base so a caller can branch on them
    without first working out which subclass it caught.

    Args:
        message: Human-readable description of the failure.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.statement: SourceCode | None = None
        self.output: str = ""
        self.has_timed_out: bool = False
        self.has_cancelled: bool = False
        # Set by integrations that own the model stream: whatever the model had
        # produced when generation was stopped.  Halting mid-response leaves the
        # provider's own message truncated, and a caller answering a tool call
        # needs something coherent to put in the transcript.
        self.partial_message: object | None = None


class ExtractionError(JitGenError):
    """The buffered source cannot become valid input for the grammar.

    Raised from `Session.push` (`jitgen.session.Session.push`) synchronously,
    so the caller can abort inference on the spot, and from the final flush.

    Args:
        message: Description of why the buffer is unrecoverable.
        source: The full buffer that failed to extract.
    """

    def __init__(self, message: str, *, source: SourceCode) -> None:
        super().__init__(message)
        self.source = source


class ExecutionError(JitGenError):
    """A dispatched statement failed at runtime.

    Args:
        message: Description of the failure.
        statement: The statement that was executing when it failed.
        output: Any stdout the statement produced before failing.
        has_timed_out: Whether the failure was due to a timeout.
        has_cancelled: Whether the statement was cancelled via `acancel()`.
    """

    def __init__(
        self,
        message: str,
        *,
        statement: SourceCode | None = None,
        output: str = "",
        has_timed_out: bool = False,
        has_cancelled: bool = False,
    ) -> None:
        super().__init__(message)
        self.statement = statement
        self.output = output
        self.has_timed_out = has_timed_out
        self.has_cancelled = has_cancelled

    @classmethod
    def from_result(
        cls, result: ExecutionResult, *, statement: SourceCode | None = None
    ) -> Self:
        """Build from a failed `ExecutionResult` (`jitgen.base.ExecutionResult`).

        Keeps `result.error` unchanged so that two callers running the same
        code through different strategies record the same string for the same
        failure, by construction rather than by convention.

        Args:
            result: The failed execution outcome to wrap.
            statement: The statement that produced `result`.

        Returns:
            An `ExecutionError` built from `result`.
        """
        return cls(
            result.error or "Execution failed",
            statement=statement,
            output=result.output or "",
            has_timed_out=result.has_timed_out,
            has_cancelled=result.has_cancelled,
        )
