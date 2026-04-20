from time import perf_counter
from uuid import uuid4

from agentic.agents.base import ExecutionResult
from agentic.prompts import format_human_prompt, format_system_prompt
from jitgen_core import BaseExecutor
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langchain.agents import create_agent


class StandardAgent:
    def __init__(
        self,
        model: BaseChatModel,
        executor: BaseExecutor,
        context_file_names: list[str],
        max_steps: int,
    ) -> None:
        self._max_steps = max_steps
        self._steps = 0

        @tool
        async def execute_code(code: str) -> str:
            """Executes Python code and returns stdout or stderr."""
            self._steps += 1
            result = await executor.aexecute(code)
            return (
                result.output or ""
                if result.success
                else f"Error detected. Halting further processing. {result.error or ''}"
            )

        self._agent = create_agent(
            model=model,
            tools=[execute_code],
            prompt=format_system_prompt(context_file_names),
        ).with_config({"recursion_limit": max_steps + 1})

    async def run(self, question: str, guidelines: str) -> ExecutionResult:
        self._steps = 0
        is_timeout = False
        thread_id = str(uuid4())

        with get_usage_metadata_callback() as cb:
            start_time = perf_counter()
            output_state = None
            try:
                output_state = await self._agent.ainvoke(
                    {"messages": [{"role": "user", "content": format_human_prompt(question, guidelines)}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
            except GraphRecursionError:
                is_timeout = True

            end_time = perf_counter()

        model_name = next(iter(cb.usage_metadata.keys()), None)
        meta = cb.usage_metadata.get(model_name, {}) if model_name else {}

        messages = output_state["messages"] if output_state else []
        last = messages[-1] if messages else None

        return ExecutionResult(
            messages=messages,
            success=not is_timeout,
            timeout=is_timeout,
            steps_taken=self._steps,
            max_steps=self._max_steps,
            output=last.content if last and not is_timeout else None,
            model_name=model_name,
            total_tokens=meta.get("total_tokens"),
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
            reasoning_tokens=meta.get("output_token_details", {}).get("reasoning"),
            total_execution_time=end_time - start_time,
        )
