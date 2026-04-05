import asyncio
import io
import sys
from collections.abc import Mapping
from types import CodeType
from typing import Any  # noqa: ANN401

from pydantic import BaseModel, Field, PrivateAttr
from jitgen_core import ExecutionResult, SourceCode


class InProcPythonExecutor(BaseModel):
    tools: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    _last_exception: BaseException | None = PrivateAttr(default=None)

    _locals: dict[str, Any] = PrivateAttr(  # noqa: ANN401
        default_factory=lambda: {"__name__": "__console__", "__doc__": None}
    )

    def model_post_init(self, __context: Any) -> None:
        self.register_tools(self.tools)

    def register_tool(self, name: str, tool: Any) -> None:  # noqa: ANN401
        self._locals[name] = tool

    def register_tools(self, tools: Mapping[str, Any]) -> None:
        for name, tool in tools.items():
            self.register_tool(name, tool)

    def _execute_code(self, compiled_code: CodeType) -> None:
        try:
            exec(compiled_code, self._locals)
        except SystemExit:
            raise
        except BaseException:
            self._last_exception = sys.exc_info()[1]
            raise

    def execute(
        self, source_code: SourceCode, *, timeout: float = 5
    ) -> ExecutionResult:
        loop = asyncio.get_event_loop()

        return loop.run_until_complete(self.aexecute(source_code, timeout=timeout))

    async def aexecute(
        self, source_code: SourceCode, *, timeout: float = 5
    ) -> ExecutionResult:
        compiled_code = compile(source_code, "<string>", "exec", optimize=2)

        stdout_collector = io.StringIO()
        stderr_collector = io.StringIO()

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = stdout_collector
        sys.stderr = stderr_collector

        has_error = False
        has_timed_out = False
        self._last_exception = None

        try:
            task = asyncio.to_thread(self._execute_code, compiled_code)

            await asyncio.wait_for(task, timeout)
        except SystemExit:
            raise
        except asyncio.TimeoutError:
            has_timed_out = True
            has_error = True
        except Exception:
            has_error = True
        finally:
            output = stdout_collector.getvalue()

            sys.stdout = original_stdout
            sys.stderr = original_stderr

            return ExecutionResult(
                success=not has_error,
                output=output,
                error=self._last_exception.__str__() if self._last_exception else None,
                has_timed_out=has_timed_out,
                timeout_value=timeout if has_timed_out else None,
            )
