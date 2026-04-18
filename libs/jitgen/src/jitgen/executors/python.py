import asyncio
import contextlib
import io
from collections.abc import Mapping
from types import CodeType
from typing import Any  # noqa: ANN401

from pydantic import BaseModel, Field, PrivateAttr

from jitgen_core import ExecutionResult, SourceCode


class InProcPythonExecutor(BaseModel):
    """In-process Python executor with REPL-style persistent state.

    ``timeout`` is owned by the executor instance; callers do not pass it
    per-call.  Instantiate with the desired value once and inject into
    :class:`~jitgen_core.Session`.

    A persistent ``_locals`` dict maintains variable state across executions
    (same semantics as a Python REPL).
    """

    timeout: float = Field(
        default=60.0,
        description="Max seconds allowed per aexecute call.",
    )
    tools: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    _locals: dict[str, Any] = PrivateAttr(
        default_factory=lambda: {
            "__name__": "__console__",
            "__doc__": None,
        }
    )

    def model_post_init(self, __context: Any) -> None:
        self.register_tools(self.tools)

    def register_tool(self, name: str, tool: Any) -> None:
        self.tools[name] = tool
        self._locals[name] = tool

    def register_tools(self, tools: Mapping[str, Any]) -> None:
        for name, tool in tools.items():
            self.register_tool(name, tool)

    def _execute_code(
        self,
        compiled_code: CodeType,
        stdout_collector: io.StringIO,
        stderr_collector: io.StringIO,
    ) -> None:
        with contextlib.redirect_stdout(stdout_collector):
            with contextlib.redirect_stderr(stderr_collector):
                exec(compiled_code, self._locals)  # noqa: S102

    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        """Execute *source_code* asynchronously, returning an :class:`~jitgen_core.ExecutionResult`.

        Uses :func:`asyncio.to_thread` so the blocking ``exec`` call does not
        stall the event loop.  The instance-level ``timeout`` is applied via
        :func:`asyncio.wait_for`.
        """
        stdout_collector = io.StringIO()
        stderr_collector = io.StringIO()
        has_error = False
        has_timed_out = False
        last_exception: BaseException | None = None
        try:
            compiled_code = compile(source_code, "<string>", "exec", optimize=2)
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._execute_code,
                    compiled_code,
                    stdout_collector,
                    stderr_collector,
                ),
                self.timeout,
            )
        except SystemExit:
            raise
        except asyncio.TimeoutError as exc:
            has_timed_out = True
            has_error = True
            last_exception = exc
        except Exception as exc:
            has_error = True
            last_exception = exc
        return ExecutionResult(
            success=not has_error,
            output=stdout_collector.getvalue() or None,
            error=str(last_exception) if last_exception is not None else None,
            has_timed_out=has_timed_out,
        )
