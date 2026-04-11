import asyncio
import contextlib
from dataclasses import dataclass
import json
import random
import re
from typing import Literal

from agentic.agents import IncrementalAgentSession, IncrementalAgentSession, SequentialAgentSession
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from jitgen.executors.python import InProcPythonExecutor
from jitgen.prebuilt.python import create_python_async_jitgen_session
from langchain_ollama import ChatOllama
from datasets import load_dataset

from langchain.agents import create_agent

from langchain.tools import tool

from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from langchain_openai import ChatOpenAI
from langsmith import traceable

import lmstudio as lms

load_dotenv(override=True)


SYSTEM_PROMPT = """
You are a helpful assistant assigned with the task of problem-solving. To achieve this, \
you will be using an interactive coding environment equipped with a variety of tool \
functions to assist you throughout the process.

After that, you have two options:
1) Interact with a Python programming environment and receive the corresponding output.
Use *execute_code* tool to run Python code. Your code should be verbatim and should not contain any markdown formatting.
2) Directly provide a solution that adheres to the required format for the given task.
Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>.
Stricly follow the *guidelines* provided when formating your solution.

## Exploring the environment:
Use *execute_code* tool to read "*.md" files to understand the data and then read "*.csv" and "*.json" files to explore the data.
For example, to read contents of a .md file, you can write:
exeute_code(```
with open("<path-to-md-file>", "r") as f:
    content = f.read()
print(content)
```)

IMPORTANT:
- Do not print content of .csv or .json files directly as it can be very large. You may print the whole content of .md files as they are usually small and contain important information about the data.
- Environment is not a Jupiter Notebook, so you should explicitly print any output you want to see.

## Available files:
You have these files available:
{context_files}

Note: *.md files contain documentation about the data, while *.csv and *.json files contain the actual data.
""".strip()

HUMAN_PROMPT = """
Here is the question you need to answer:
```
{question}
```

Here are the guidelines you must STRICTLY follow when answering the question above:
```
{guidelines}
```
"""
question = "What are the unique set of merchants in the payments data?"
guidelines = "Answer with a comma separated list"




def load_context_files() -> list[str]:
    CONTEXT_FILENAMES = [
        "data/context/acquirer_countries.csv",
        "data/context/payments-readme.md",
        "data/context/payments.csv",
        "data/context/merchant_category_codes.csv",
        "data/context/fees.json",
        "data/context/merchant_data.json",
        "data/context/manual.md",
    ]

    DATA_DIR = "./tmp/DABstep-data"
    
    for filename in CONTEXT_FILENAMES:
        hf_hub_download(
            repo_id="adyen/DABstep",
            repo_type="dataset",
            filename=filename,
            local_dir=DATA_DIR,
            # force_download=True
        )

    CONTEXT_FILENAMES = [f"{DATA_DIR}/{filename}" for filename in CONTEXT_FILENAMES]
    
    return CONTEXT_FILENAMES

def load_dataset_files() -> list[dict[str, str]]:
    dataset = load_dataset(
        "adyen/DABstep",
    )

    return dataset["dev"]


async def restart_model(model_name: str):
    await (ChatOllama(
        model=model_name,
        temperature=0,
        reasoning=False,
        keep_alive=0,
        num_predict=2,
    ).ainvoke("Hi"))




@traceable
async def incremental_agent_session(model: BaseChatModel, context_file_names: list[str], question: str, guidelines: str, max_steps: int):
    @tool
    def execute_code(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        return ""
    
    session = create_python_async_jitgen_session(
        start_marker="{\"code\":\"", end_marker="\"}", tools={"open": open}
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
        {"role": "system", "content": SYSTEM_PROMPT.format(context_files="\n".join(context_file_names))},
        {"role": "user", "content": HUMAN_PROMPT.format(question=question, guidelines=guidelines)},
    ]
    
    for step in range(max_steps):
        ai_message: AIMessage | None = None
        tool_call_id = None
        response = ""
        is_error = False
        ai_content = ""
        tool_call_name = ""
        raw_args = ""
        async for event in _model.astream_events(messages):
            if is_error:
                break
            
            if event["event"] == "on_llm_end" or event["event"] == "on_chat_model_end":
                pass
            
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
                        await session.apush(tool_chunk["args"].encode().decode('unicode_escape'))
            elif event["event"] == "on_chat_model_end":
                ai_message = event['data']['output']
            else:
                pass
        
        try:
            if not is_error:
                await session.aflush()
        except Exception as e:
            if not is_error:
                raise
        finally:
            if is_error:
                session._algorithm._code_buffer = "" # Clear any buffered code in the session to prevent it from being executed in the next step
                session._algorithm._raw_buffer = "" # Clear any raw buffered input as well

            session._algorithm._inside_markers = False # Reset marker state after each step to allow for new code blocks in subsequent steps FIXME: Expose a proper API for this in the algorithm/session instead of reaching into internals
            # self._handled_session_error = None
        
        if ai_message and len(ai_message.tool_calls) == 0:
            break
        
        if ai_message is None:
            ai_message = AIMessage(content=ai_content, tool_calls=[
                {
                    "id": tool_call_id, 
                    "name": tool_call_name,
                    "args": {
                        "code": json.loads((raw_args + "\"}") if not raw_args.endswith("}") else raw_args)
                    }, # We already pushed the tool args to the session as they came in, so we can leave this empty to avoid confusion
                }
            ]) # Create an AIMessage with the accumulated content if we didn't get a proper message from the model, to ensure we can at least return any output received before an error occurred
        messages.append(ai_message)
        messages.append(ToolMessage(content=response if raw_args not in ("", "{}") else "\"code\" should be provided.", tool_call_id=tool_call_id, name=tool_call_name))
    
    return ai_message.content


@traceable
async def sequential_agent_session(model: BaseChatModel, context_file_names: list[str], question: str, guidelines: str, max_steps: int):
    @tool
    def execute_code(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        return ""
    executor = InProcPythonExecutor()
    

    async def _execute_code_impl(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        nonlocal executor
        result = await executor.aexecute(code)

        return(
            result.output or ""
            if result.success
            else ("Error detected. Halting further processing. " + (result.error or ""))
        )
    
    _model = model.bind_tools([execute_code])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context_files="\n".join(context_file_names))},
        {"role": "user", "content": HUMAN_PROMPT.format(question=question, guidelines=guidelines)},
    ]
    
    for step in range(max_steps):
        ai_message: AIMessage | None = None
        tool_call_id = None
        response = ""
        is_error = False
        ai_content = ""
        tool_call_name = ""
        raw_args = ""
        async for event in _model.astream_events(messages):
            if is_error:
                break
            
            if event["event"] == "on_llm_end" or event["event"] == "on_chat_model_end":
                pass
            
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
                ai_message = event['data']['output']
            else:
                pass
        
        if ai_message and len(ai_message.tool_calls) == 0:
            break
        
        if raw_args not in ("", "{}"):
            try:
                execution_result = await _execute_code_impl(json.loads(raw_args).get("code", ""))
                response = execution_result
            except Exception as e:
                response = "Error detected. Halting further processing. " + str(e)
                is_error = True
        
        messages.append(ai_message)
        messages.append(ToolMessage(content=response if raw_args not in ("", "{}") else "\"code\" should be provided.", tool_call_id=tool_call_id, name=tool_call_name))
    
    return ai_message.content


@traceable
async def langchain_agent_session(model: BaseChatModel, context_file_names: list[str], question: str, guidelines: str, max_steps: int):
    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT.format(context_files="\n".join(context_file_names))},
    #     
    # ]
    executor = InProcPythonExecutor()
    
    @tool
    async def execute_code(code: str) -> str:
        """Executes Python code and returns stdout or stderr."""
        nonlocal executor
        result = await executor.aexecute(code)

        return(
            result.output or ""
            if result.success
            else ("Error detected. Halting further processing. " + (result.error or ""))
        )
    
    
    agent = create_agent(
        model=model,
        tools=[execute_code],
        system_prompt=SYSTEM_PROMPT.format(context_files="\n".join(context_file_names)),
    ).with_config({"recursion_limit": max_steps})
    
    output_state = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": HUMAN_PROMPT.format(question=question, guidelines=guidelines)},
        ]
    })
    
    
    return output_state["messages"][-1].content


async def main():
    model_name = "openai/gpt-oss-20b" # "google/gemma-4-e4b"
    mode: Literal["sequential", "incremental"] = "incremental"
    max_steps = 20
    run_timeout_seconds = 30.0
    
    session_id = 1234
    temperature = 0.3
        

    
    
    # await restart_model(model_name)
    # model = ChatOllama(
    #     model=model_name,
    #     temperature=0,
    #     seed=session_id,
    #     # reasoning=True,
    # )
    
    model = ChatOpenAI(
        model=model_name,
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0,
        stream_usage=True,
        streaming=True,
    )
    
    context_file_names = load_context_files()
    dataset = load_dataset_files()

    test_case = dataset[7]
    async with lms.AsyncClient() as client:
        with contextlib.suppress(lms.LMStudioModelNotFoundError):
            await client.llm.unload(model_name)
        await client.llm.load_new_instance(model_name, config=lms.LlmLoadModelConfig(
            seed=session_id,
        ))
    
    start_time = perf_counter()
    response = await sequential_agent_session(
        model=model,
        context_file_names=context_file_names,
        question=test_case["question"],
        guidelines=test_case["guidelines"],
        max_steps=max_steps,
    )
    
    print("Task: ", test_case["question"])
    print("Final response:", response)
    print("Correct answer:", test_case["answer"])
    print(f"Total execution time: {perf_counter() - start_time:.2f} seconds")
    
    
    async with lms.AsyncClient() as client:
        with contextlib.suppress(lms.LMStudioModelNotFoundError):
            await client.llm.unload(model_name)
        await client.llm.load_new_instance(model_name, config=lms.LlmLoadModelConfig(
            seed=session_id,
        ))
    
    
    
    start_time = perf_counter()
    response = await incremental_agent_session(
        model=model,
        context_file_names=context_file_names,
        question=test_case["question"],
        guidelines=test_case["guidelines"],
        max_steps=max_steps,
    )
    
    print("Task: ", test_case["question"])
    print("Final response:", response)
    print("Correct answer:", test_case["answer"])
    print(f"Total execution time: {perf_counter() - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())



