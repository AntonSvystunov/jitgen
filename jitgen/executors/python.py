import asyncio
import io
import sys
from types import CodeType
from typing import Any

from pydantic import BaseModel, PrivateAttr
from jitgen.core.base import ExecutionResult, SourceCode


class InProcPythonExecutor(BaseModel):
    _last_exception: Exception | None = PrivateAttr(default=None)

    _locals: dict[str, Any] = PrivateAttr(
        default_factory=lambda: {"__name__": "__console__", "__doc__": None}
    )

    def _execute_code(self, compiled_code: CodeType) -> None:
        try:
            exec(compiled_code, self._locals)
        except SystemExit:
            raise
        except:
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
