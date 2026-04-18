import asyncio
import json

from agentic.agents import LangchainAgent
from dotenv import load_dotenv
from jitgen.executors.python import InProcPythonExecutor
from jitgen.markers import MarkerStripper
from jitgen.prebuilt.python import create_python_jitgen

from langchain.agents import create_agent

from langchain.tools import tool

from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from langchain_openai import ChatOpenAI
from langsmith import traceable

from langgraph.errors import GraphRecursionError
from langchain_core.callbacks import get_usage_metadata_callback

from agentic.data import load_context_files, load_tasks_dataset
from agentic.models import get_model
from agentic.prompts import (
    format_system_prompt,
    format_human_prompt,
    format_execute_code_tool_error,
)

load_dotenv(override=True)


@traceable
async def incremental_agent_session(
    model: ChatOpenAI,
    context_file_names: list[str],
    question: str,
    guidelines: str,
    max_steps: int,
):
    @tool
    def execute_code(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        return ""

    executor = InProcPythonExecutor(tools={"open": open})
    session = create_python_jitgen(executor=executor)
    stripper = MarkerStripper(start='{"code":"', end='"}')

    _model = model.bind_tools([execute_code])

    messages = [
        {"role": "system", "content": format_system_prompt(context_file_names)},
        {"role": "user", "content": format_human_prompt(question, guidelines)},
    ]

    for step in range(max_steps):
        session.reset()
        stripper.reset()

        ai_message: AIMessage | None = None
        tool_call_id = None
        ai_content = ""
        tool_call_name = ""
        raw_args = ""
        execution_started: float | None = None
        step_start_time = perf_counter()

        async for event in _model.astream_events(messages):
            if session.has_error:
                break

            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    ai_content += chunk.content
                for tool_chunk in chunk.tool_call_chunks:
                    if tool_chunk.get("id"):
                        tool_call_id = tool_chunk["id"]
                    if tool_chunk.get("name"):
                        tool_call_name = tool_chunk["name"]
                    if tool_chunk.get("args"):
                        raw_args += tool_chunk["args"]
                        if not execution_started:
                            execution_started = perf_counter()
                        decoded = tool_chunk["args"].encode().decode("unicode_escape")
                        for seg in stripper.process(decoded):
                            session.push(seg.text)
                            if session.has_error:
                                break
            elif event["event"] == "on_chat_model_end":
                ai_message = event["data"]["output"]

        inference_time = perf_counter() - step_start_time

        is_error = session.has_error
        if is_error:
            response = str(session.error)
        else:
            try:
                response = await session.result()
                is_error = False
            except Exception as exc:
                response = str(exc)
                is_error = True

        step_total_execution_time = (
            perf_counter() - execution_started if execution_started else 0
        )
        step_end_time = perf_counter()

        if ai_message and len(ai_message.tool_calls) == 0:
            break

        if ai_message is None:
            ai_message = AIMessage(
                content=ai_content,
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": tool_call_name,
                        "args": json.loads(
                            (raw_args + '"}')
                            if not raw_args.endswith("}")
                            else raw_args
                        ),
                    }
                ],
            )

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

        if is_error:
            tool_message_content = format_execute_code_tool_error(response)
        else:
            tool_message_content = (
                response if raw_args not in ("", "{}") else '"code" should be provided.'
            )

        messages.append(
            ToolMessage(
                content=tool_message_content,
                tool_call_id=tool_call_id,
                name=tool_call_name,
            )
        )

    return ai_message.content


@traceable
async def sequential_agent_session(
    model: ChatOpenAI,
    context_file_names: list[str],
    question: str,
    guidelines: str,
    max_steps: int,
):
    @tool
    def execute_code(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        return ""

    executor = InProcPythonExecutor()

    async def _execute_code_impl(code: str) -> str:
        result = await executor.aexecute(code)
        return (
            result.output or ""
            if result.success
            else format_execute_code_tool_error(result.error or "")
        )

    _model = model.bind_tools([execute_code])

    messages = [
        {"role": "system", "content": format_system_prompt(context_file_names)},
        {"role": "user", "content": format_human_prompt(question, guidelines)},
    ]

    for step in range(max_steps):
        ai_message: AIMessage | None = None
        tool_call_id = None
        response = ""
        is_error = False
        ai_content = ""
        tool_call_name = ""
        raw_args = ""
        execution_started: float | None = None
        step_start_time = perf_counter()

        async for event in _model.astream_events(messages):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    ai_content += chunk.content
                for tool_chunk in chunk.tool_call_chunks:
                    if tool_chunk.get("id"):
                        tool_call_id = tool_chunk["id"]
                    if tool_chunk.get("name"):
                        tool_call_name = tool_chunk["name"]
                    if tool_chunk.get("args"):
                        raw_args += tool_chunk["args"]
            elif event["event"] == "on_chat_model_end":
                ai_message = event["data"]["output"]

        inference_time = perf_counter() - step_start_time

        if ai_message and len(ai_message.tool_calls) == 0:
            break

        if raw_args not in ("", "{}"):
            try:
                execution_started = perf_counter()
                response = await _execute_code_impl(
                    json.loads(raw_args).get("code", "")
                )
            except Exception as exc:
                response = "Error detected. Halting further processing. " + str(exc)
                is_error = True

        step_total_execution_time = (
            perf_counter() - execution_started if execution_started else 0
        )
        step_end_time = perf_counter()

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
        messages.append(
            ToolMessage(
                content=response
                if raw_args not in ("", "{}")
                else '"code" should be provided.',
                tool_call_id=tool_call_id,
                name=tool_call_name,
            )
        )

    return ai_message.content


@traceable
async def langchain_agent_session(
    model: ChatOpenAI,
    context_file_names: list[str],
    question: str,
    guidelines: str,
    max_steps: int,
):
    executor = InProcPythonExecutor()

    @tool
    async def execute_code(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        result = await executor.aexecute(code)
        return (
            result.output or ""
            if result.success
            else ("Error detected. Halting further processing. " + (result.error or ""))
        )

    agent = create_agent(
        model=model,
        tools=[execute_code],
        system_prompt=format_system_prompt(context_file_names),
    ).with_config({"recursion_limit": max_steps})

    try:
        output_state = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": format_human_prompt(question, guidelines),
                    }
                ]
            }
        )
    except GraphRecursionError:
        return ""

    return output_state["messages"][-1].content


async def main():
    context_file_names = load_context_files("./tmp")
    dataset = load_tasks_dataset("dev")

    model_name = "google/gemma-4-e4b"
    seed = 1234
    temperature = 0.7

    async with get_model(
        model_name=model_name, temperature=temperature, seed=seed
    ) as model:
        for test_case in dataset.skip(5).take(2):
            print("Task: ", test_case["task_id"])

            response = await incremental_agent_session(
                model=model,
                context_file_names=context_file_names,
                question=test_case["question"],
                guidelines=test_case["guidelines"],
                max_steps=20,
            )

            print("Final result: ", response)
    
    async with get_model(
        model_name=model_name, temperature=temperature, seed=seed
    ) as model:
        for test_case in dataset.skip(5).take(2):
            print("Task: ", test_case["task_id"])

            response = await langchain_agent_session(
                model=model,
                context_file_names=context_file_names,
                question=test_case["question"],
                guidelines=test_case["guidelines"],
                max_steps=20,
            )

            print("Final result: ", response)


if __name__ == "__main__":
    asyncio.run(main())
