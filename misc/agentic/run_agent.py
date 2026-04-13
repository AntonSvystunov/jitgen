import asyncio
import json
from uuid import uuid4

from agentic.results import ResultsLogger
from dotenv import load_dotenv
from jitgen.executors.python import InProcPythonExecutor
from jitgen.prebuilt.python import create_python_async_jitgen_session

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
    results_logger: ResultsLogger,
):
    @tool
    def execute_code(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        return ""

    session = create_python_async_jitgen_session(
        start_marker='{"code":"', end_marker='"}', tools={"open": open}
    )

    @session.on_stdout
    def on_stdout(stdout: str):
        nonlocal response
        nonlocal is_error

        if not is_error:
            response += stdout

    @session.on_error
    def on_error(stderr: Exception):
        nonlocal response
        nonlocal is_error
        if not is_error:
            response = str(stderr)
        is_error = True

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
            if is_error:
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
                        await session.apush(
                            tool_chunk["args"].encode().decode("unicode_escape")
                        )
            elif event["event"] == "on_chat_model_end":
                ai_message = event["data"]["output"]
            else:
                pass
        inference_time = perf_counter() - step_start_time

        try:
            if not is_error:
                await session.aflush()
        except Exception:
            if not is_error:
                raise
        finally:
            step_total_execution_time = perf_counter() - execution_started if execution_started else 0
            if is_error:
                session._algorithm._code_buffer = ""  # Clear any buffered code in the session to prevent it from being executed in the next step
                session._algorithm._raw_buffer = (
                    ""  # Clear any raw buffered input as well
                )

            session._algorithm._inside_markers = False  # Reset marker state after each step to allow for new code blocks in subsequent steps FIXME: Expose a proper API for this in the algorithm/session instead of reaching into internals
            # self._handled_session_error = None
        step_end_time = perf_counter()

        results_logger.add_step(
            step_number=step + 1,
            message=ai_message.content if ai_message else None,
            code=raw_args if raw_args not in ("", "{}") else None,
            observation=response if raw_args not in ("", "{}") else None,
            is_error=is_error,
            tool_call_success=not is_error if raw_args not in ("", "{}") else None,
            tool_call_error_message=response if is_error else None,
            tool_execution_time=step_total_execution_time if raw_args not in ("", "{}") else None,
            inference_time=inference_time if raw_args not in ("", "{}") else None,
            total_time=step_end_time - step_start_time
        )
        
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
                        ),  # We already pushed the tool args to the session as they came in, so we can leave this empty to avoid confusion
                    }
                ],
            )  # Create an AIMessage with the accumulated content if we didn't get a proper message from the model, to ensure we can at least return any output received before an error occurred

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

    results_logger.set_messages(messages)

    return ai_message.content


@traceable
async def sequential_agent_session(
    model: ChatOpenAI,
    context_file_names: list[str],
    question: str,
    guidelines: str,
    max_steps: int,
    results_logger: ResultsLogger,
):
    @tool
    def execute_code(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        return ""

    executor = InProcPythonExecutor()

    async def _execute_code_impl(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        nonlocal executor
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
            if is_error:
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
            elif event["event"] == "on_chat_model_end":
                ai_message = event["data"]["output"]
            else:
                pass
        
        inference_time = perf_counter() - step_start_time
        
        if ai_message and len(ai_message.tool_calls) == 0:
            break

        if raw_args not in ("", "{}"):
            try:
                execution_started = perf_counter()
                execution_result = await _execute_code_impl(
                    json.loads(raw_args).get("code", "")
                )
                response = execution_result
            except Exception as e:
                response = "Error detected. Halting further processing. " + str(e)
                is_error = True

        step_total_execution_time = perf_counter() - execution_started if execution_started else 0
        step_end_time = perf_counter()
        
        results_logger.add_step(
            step_number=step + 1,
            message=ai_message.content if ai_message else None,
            code=raw_args if raw_args not in ("", "{}") else None,
            observation=response if raw_args not in ("", "{}") else None,
            is_error=is_error,
            tool_call_success=not is_error if raw_args not in ("", "{}") else None,
            tool_call_error_message=response if is_error else None,
            tool_execution_time=step_total_execution_time if raw_args not in ("", "{}") else None,
            inference_time=inference_time if raw_args not in ("", "{}") else None,
            total_time=step_end_time - step_start_time
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
        nonlocal executor
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
    # FIXME: handle graph recursion error.
    
    try:
        output_state = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": format_human_prompt(question, guidelines)},
                ]
            }
        )
    except GraphRecursionError:
        return ""

    return output_state["messages"][-1].content


async def main():
    context_file_names = load_context_files("./tmp")
    dataset = load_tasks_dataset("dev")

    model_name = "google/gemma-4-e4b" # "openai/gpt-oss-20b"  # 
    seed = 1234
    temperature = 0

    test_case = dataset[1]  # 7

    run_id = uuid4()
    
    incremental_steps = []
    sequential_steps = []

    async with get_model(
        model_name=model_name, temperature=temperature, seed=seed
    ) as model:
        start_time = perf_counter()
        
        results_logger = ResultsLogger(run_id, model_name, temperature, seed, type="incremental")
        
        with get_usage_metadata_callback() as cb:
            response = await incremental_agent_session(
                model=model,
                context_file_names=context_file_names,
                question=test_case["question"],
                guidelines=test_case["guidelines"],
                max_steps=20,
                results_logger=results_logger,
            )
            
            print("Usage metadata:", cb.usage_metadata)
            
            for step in results_logger.steps:
                print(f"Step {step.step_number}:")
                print("Message:", step.message)
                print("Code:", step.code)
                # print("Observation:", step.observation)
                print("Is error:", step.is_error)
                print("Tool call success:", step.tool_call_success)
                print("Tool call error message:", step.tool_call_error_message)
                print("Inference time:", step.inference_time)
                print("Tool execution time:", step.tool_execution_time)
                print("Total time:", step.total_time)
                print("-" * 20)
                
            incremental_steps = results_logger.steps

        print("Task: ", test_case["question"])
        print("Final response:", response)
        print("Correct answer:", test_case["answer"])
        print(f"Total execution time: {perf_counter() - start_time:.2f} seconds")

    async with get_model(
        model_name=model_name, temperature=temperature, seed=seed
    ) as model:
        start_time = perf_counter()
        results_logger = ResultsLogger(run_id, model_name, temperature, seed, type="sequential")
        response = await sequential_agent_session(
            model=model,
            context_file_names=context_file_names,
            question=test_case["question"],
            guidelines=test_case["guidelines"],
            max_steps=20,
            results_logger=results_logger,
        )
        
        for step in results_logger.steps:
            print(f"Step {step.step_number}:")
            print("Message:", step.message)
            print("Code:", step.code)
            # print("Observation:", step.observation)
            print("Is error:", step.is_error)
            print("Tool call success:", step.tool_call_success)
            print("Tool call error message:", step.tool_call_error_message)
            print("Inference time:", step.inference_time)
            print("Tool execution time:", step.tool_execution_time)
            print("Total time:", step.total_time)
            print("-" * 20)
            
        sequential_steps = results_logger.steps

        print("Task: ", test_case["question"])
        print("Final response:", response)
        print("Correct answer:", test_case["answer"])
        print(f"Total execution time: {perf_counter() - start_time:.2f} seconds")


    for inc, seq in zip(incremental_steps, sequential_steps):
        print(f"Step {inc.step_number}:")
        print("Inference time difference:", inc.inference_time - seq.inference_time if inc.inference_time and seq.inference_time else None)
        print("Tool execution time difference:", inc.tool_execution_time - seq.tool_execution_time if inc.tool_execution_time and seq.tool_execution_time else None)
        print("Total time difference:", inc.total_time - seq.total_time)
        
        print("-" * 20)


    return

    async with get_model(
        model_name=model_name, temperature=temperature, seed=seed
    ) as model:
        start_time = perf_counter()
        
        response = await langchain_agent_session(
            model=model,
            context_file_names=context_file_names,
            question=test_case["question"],
            guidelines=test_case["guidelines"],
            max_steps=20,
        )

        print("Task: ", test_case["question"])
        print("Final response:", response)
        print("Correct answer:", test_case["answer"])
        print(f"Total execution time: {perf_counter() - start_time:.2f} seconds")

   


if __name__ == "__main__":
    asyncio.run(main())
