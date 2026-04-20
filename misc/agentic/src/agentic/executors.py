import asyncio
from datetime import timedelta
from typing import Any, Self

from code_interpreter import CodeInterpreter, SupportedLanguage
from jitgen_core import ExecutionResult, SourceCode
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.execd import ExecutionHandlers
from pydantic import BaseModel, Field, PrivateAttr

from langsmith import traceable


class OpenSandboxPythonExecutor(BaseModel):
    """Remote Python executor backed by an OpenSandbox code-interpreter container.

    Sandbox, interpreter, and Python context are created lazily on the first
    ``aexecute()`` call and shared across all subsequent calls (REPL-style state).
    Use as an async context manager to ensure the sandbox is cleaned up:

        async with OpenSandboxPythonExecutor() as executor:
            result = await executor.aexecute("print('hello')")
    """

    image: str = Field(default="agentic-opensandbox-code-interpreter:dabstep")
    domain: str = Field(default="localhost:8080")
    request_timeout_seconds: float = Field(default=120.0)
    packages: list[str] = Field(default_factory=list)

    _sandbox: Sandbox | None = PrivateAttr(default=None)
    _interpreter: CodeInterpreter | None = PrivateAttr(default=None)
    _context: Any | None = PrivateAttr(default=None)
    _packages_installed: bool = PrivateAttr(default=False)
    _execution_id: str | None = PrivateAttr(default=None)

    async def _ensure_setup(self) -> None:
        if self._sandbox is not None and self._context is not None:
            return

        if self._sandbox is None:
            config = ConnectionConfig(
                domain=self.domain,
                request_timeout=timedelta(seconds=self.request_timeout_seconds),
            )
            self._sandbox = await Sandbox.create(
                self.image,
                connection_config=config,
                entrypoint=["/opt/opensandbox/code-interpreter.sh"],
                env={
                    "PYTHON_VERSION": "3.11",
                },
            )
            self._interpreter = await CodeInterpreter.create(sandbox=self._sandbox)

        if self._context is None:
            self._context = await self._interpreter.codes.create_context(
                SupportedLanguage.PYTHON
            )

        if self.packages and not self._packages_installed:
            pkg_list = " ".join(self.packages)
            result = await self._interpreter.codes.run(
                f"%pip install {pkg_list} --break-system-packages",
                context=self._context,
            )
            if result.error:
                raise RuntimeError(f"Package installation failed: {result.error}")
            self._packages_installed = True

    @traceable(run_type="tool")
    async def aexecute(self, source_code: SourceCode) -> ExecutionResult:
        await self._ensure_setup()

        self._execution_id = None

        async def on_init(msg: Any) -> None:
            self._execution_id = getattr(msg, "id", None)

        handlers = ExecutionHandlers(on_init=on_init)

        has_timed_out = False
        last_exception: BaseException | None = None
        execution = None

        try:
            execution = await asyncio.wait_for(
                self._interpreter.codes.run(
                    source_code + "\n",
                    context=self._context,
                    handlers=handlers,
                ),
                self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            has_timed_out = True
            last_exception = exc
            await self.acancel()
        except asyncio.CancelledError:
            await self.acancel()
            raise
        except Exception as exc:
            last_exception = exc

        has_error = last_exception is not None or (
            execution is not None and bool(execution.error)
        )
        error_msg: str | None = None
        if has_timed_out:
            error_msg = (
                f"Execution timed out after {self.request_timeout_seconds:g}s. "
                "The sandbox session was interrupted and reset; previous REPL "
                "state (variables, imports) has been lost."
            )
        elif last_exception is not None:
            error_msg = str(last_exception)
        elif execution is not None and execution.error:
            err = execution.error
            name = getattr(err, "name", None)
            value = getattr(err, "value", None)
            if name or value:
                error_msg = f"{name or ''}: {value or ''}".strip(": ")
            else:
                error_msg = str(err)

        output: str | None = None
        if execution is not None:
            raw = "\n".join(msg.text for msg in execution.logs.stdout)
            output = raw or None

        return ExecutionResult(
            success=not has_error,
            output=output,
            error=error_msg,
            has_timed_out=has_timed_out,
        )

    async def acancel(self) -> None:
        """Best-effort abort of any in-flight execution.

        Attempts to interrupt the running sandbox execution (if an execution ID
        is known) then replaces the Python context so the next ``aexecute``
        starts with a clean session.  REPL state from the previous context is
        lost.
        """
        if self._interpreter is None:
            return
        if self._execution_id is not None:
            try:
                await asyncio.wait_for(
                    self._interpreter.codes.interrupt(self._execution_id),
                    timeout=5.0,
                )
            except Exception:
                pass
            self._execution_id = None
        await self._reset_context()

    async def _reset_context(self) -> None:
        """Replace the Python execution context with a fresh one."""
        if self._interpreter is None:
            return
        old_context = self._context
        try:
            self._context = await self._interpreter.codes.create_context(
                SupportedLanguage.PYTHON
            )
        except Exception:
            self._context = None
        if old_context is not None and old_context.id is not None:
            try:
                await self._interpreter.codes.delete_context(old_context.id)
            except Exception:
                pass

    async def aclose(self) -> None:
        if self._sandbox is not None:
            await self._sandbox.kill()
            self._sandbox = None
            self._interpreter = None
            self._context = None
            self._packages_installed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
