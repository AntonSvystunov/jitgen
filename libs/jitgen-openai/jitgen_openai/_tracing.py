from collections.abc import Awaitable, Callable
from typing import Any

from langsmith.run_trees import RunTree


class EvalTracer:
    """Traces each turn's `eval` execution, and the tool calls made from it.

    The executed code calls tools from the executor's worker thread, where the
    LangSmith context of the agent run is not visible, so every child run is
    created from an explicit parent instead.

    Args:
        parent: The agent run to attach `eval` runs to, or `None` to disable
            tracing.
        language: The language of the executed code, recorded on each `eval`
            run so LangSmith renders it with the right syntax.
    """

    def __init__(self, parent: RunTree | None, language: str = "python") -> None:
        self._parent = parent
        self._language = language
        self._eval_run: RunTree | None = None

    def start_eval(self) -> None:
        """Open this turn's `eval` run, if tracing and not already open."""
        if self._parent is None or self._eval_run is not None:
            return
        run = self._parent.create_child(
            name="eval",
            run_type="tool",
            inputs={},
            extra={"metadata": {"ls_code_input_language": self._language}},
        )
        run.post()
        self._eval_run = run

    def end_eval(self, code: str, output: str, error: str | None) -> None:
        """Close this turn's `eval` run with its code and result.

        Args:
            code: The executed code, known in full only once the turn ends.
            output: The tool result sent back to the model.
            error: The execution error, if any.
        """
        run, self._eval_run = self._eval_run, None
        if run is None:
            return
        run.inputs = {"code": code}
        run.end(outputs={"output": output}, error=error)
        run.patch(exclude_inputs=False)

    def wrap_tool(
        self, name: str, tool: Callable[..., Awaitable[Any]]
    ) -> Callable[..., Awaitable[Any]]:
        """Wrap `tool` so each call is traced under the open `eval` run.

        Args:
            name: The tool name, used as the run name.
            tool: The async tool callable exposed to the executed code.

        Returns:
            An async callable with the same signature as `tool`.
        """

        async def call(*args: Any, **kwargs: Any) -> Any:
            parent = self._eval_run
            if parent is None:
                return await tool(*args, **kwargs)
            inputs = {**kwargs, "args": list(args)} if args else kwargs
            run = parent.create_child(name=name, run_type="tool", inputs=inputs)
            run.post()
            try:
                output = await tool(*args, **kwargs)
            except BaseException as exc:
                run.end(error=repr(exc))
                raise
            else:
                run.end(outputs={"output": output})
                return output
            finally:
                run.patch()

        call.__name__ = name
        return call
