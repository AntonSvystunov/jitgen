from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from jitgen.base import ExecutionResult, SourceCode


class ExecutorBase(ABC):
    """Implement `aexecute` and get the rest of the contract for free.

    `BaseExecutor` (`jitgen.base.BaseExecutor`) requires `acancel` and
    `aclose` as well, because a session must be able to interrupt in-flight
    work on reset and release resources on teardown. Most executors have
    nothing to do for either, so they are no-ops here — an executor that
    *can* interrupt (a thread, a remote sandbox) overrides them.

    Also an async context manager, so ownership is expressible:

    ```python
    async with InProcPythonExecutor(timeout=10) as executor:
        ...
    ```
    """

    @abstractmethod
    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        """Run `source_code` and report what happened.

        Args:
            source_code: One dispatched top-level statement.

        Returns:
            The outcome of running `source_code`.
        """

    async def acancel(self) -> None:
        """Interrupt the statement currently executing, if any."""

    async def aclose(self) -> None:
        """Release resources held by this executor."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
