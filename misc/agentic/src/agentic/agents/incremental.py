import json
from time import perf_counter

from agentic.agents.base import ExecutionResult
from agentic.prompts import (
    format_execute_code_tool_error,
    format_human_prompt,
    format_system_prompt,
)
from jitgen_core import BaseExecutor
from jitgen.markers import MarkerStripper
from jitgen.prebuilt.python import create_python_jitgen
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langsmith import traceable


class IncrementalAgent:
    def __init__(
        self,
        model: BaseChatModel,
        executor: BaseExecutor,
        context_file_names: list[str],
        max_steps: int,
    ) -> None:
        self._max_steps = max_steps
        self._context_file_names = context_file_names

        @tool
        def execute_code(code: str) -> str:
            """Executes Python code and returns stdout or stderr."""
            return ""

        self._session = create_python_jitgen(executor=executor)
        self._stripper = MarkerStripper(start='{"code":"', end='"}')
        self._tool = execute_code
        self._model = model.bind_tools([execute_code])

    @traceable(name="IncrementalAgent.run")
    async def run(self, question: str, guidelines: str) -> ExecutionResult:
        messages: list = [
            {"role": "system", "content": format_system_prompt(self._context_file_names)},
            {"role": "user", "content": format_human_prompt(question, guidelines)},
        ]

        steps_taken = 0
        total_inference_time = 0.0
        total_execution_time_accum = 0.0
        any_early_exit = False
        last_ai_message: AIMessage | None = None

        with get_usage_metadata_callback() as cb:
            overall_start = perf_counter()

            for _ in range(self._max_steps):
                await self._session.reset()
                self._stripper.reset()

                ai_message: AIMessage | None = None
                tool_call_id: str | None = None
                tool_call_name = ""
                ai_content = ""
                raw_args = ""
                execution_started: float | None = None
                step_start = perf_counter()

                async for event in self._model.astream_events(messages):
                    if self._session.has_error:
                        any_early_exit = True
                        break

                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            ai_content += chunk.content
                        for tc in chunk.tool_call_chunks:
                            if tc.get("id"):
                                tool_call_id = tc["id"]
                            if tc.get("name"):
                                tool_call_name = tc["name"]
                            if tc.get("args"):
                                raw_args += tc["args"]
                                if execution_started is None:
                                    execution_started = perf_counter()
                                decoded = tc["args"].encode().decode("unicode_escape")
                                for seg in self._stripper.process(decoded):
                                    self._session.push(seg.text)
                                    if self._session.has_error:
                                        break

                    elif event["event"] == "on_chat_model_end":
                        ai_message = event["data"]["output"]

                inference_end = perf_counter()
                total_inference_time += inference_end - step_start

                # Always drain pending executions even on error to avoid wedging the sandbox.
                try:
                    response = await self._session.result()
                    is_error = False
                except Exception as exc:
                    response = str(exc)
                    is_error = True

                if execution_started is not None:
                    total_execution_time_accum += perf_counter() - execution_started

                # No tool call → model is done
                if ai_message is not None and len(ai_message.tool_calls) == 0:
                    last_ai_message = ai_message
                    break

                # Build AIMessage manually when stream was cut early
                if ai_message is None:
                    try:
                        parsed_args = json.loads(
                            (raw_args + '"}') if not raw_args.endswith("}") else raw_args
                        )
                    except Exception:
                        parsed_args = {}
                    ai_message = AIMessage(
                        content=ai_content,
                        tool_calls=[
                            {"id": tool_call_id, "name": tool_call_name, "args": parsed_args}
                        ] if tool_call_id else [],
                    )

                last_ai_message = ai_message
                steps_taken += 1

                messages.append(
                    AIMessage(
                        content=ai_message.content,
                        additional_kwargs=ai_message.additional_kwargs,
                        response_metadata=ai_message.response_metadata,
                        usage_metadata=ai_message.usage_metadata,
                        tool_calls=ai_message.tool_calls,
                        id=ai_message.id,
                    )
                )

                tool_content = (
                    format_execute_code_tool_error(response)
                    if is_error
                    else (response if raw_args not in ("", "{}") else '"code" should be provided.')
                )
                messages.append(
                    ToolMessage(
                        content=tool_content,
                        tool_call_id=tool_call_id,
                        name=tool_call_name,
                    )
                )

            overall_end = perf_counter()

        model_name = next(iter(cb.usage_metadata.keys()), None)
        meta = cb.usage_metadata.get(model_name, {}) if model_name else {}

        # Collect all AnyMessage objects from message history (skip plain dicts)
        collected_messages = [m for m in messages if isinstance(m, (AIMessage, ToolMessage))]

        return ExecutionResult(
            messages=collected_messages,
            success=True,
            timeout=steps_taken >= self._max_steps,
            steps_taken=steps_taken,
            max_steps=self._max_steps,
            output=last_ai_message.content if last_ai_message else None,
            model_name=model_name,
            total_tokens=meta.get("total_tokens"),
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
            reasoning_tokens=meta.get("output_token_details", {}).get("reasoning"),
            total_execution_time=overall_end - overall_start,
            inference_time=total_inference_time,
            execution_time=total_execution_time_accum,
            early_exit_on_error=any_early_exit,
        )
