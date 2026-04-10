import asyncio
from dataclasses import dataclass
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

from langchain.tools import tool

from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from langchain_openai import ChatOpenAI
from langsmith import traceable

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
- Do not print content of .csv or .json files directly as it can be very large. Instead, use Python to explore the data and print only relevant information.
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


@tool
def execute_code(code: str) -> str:
    """Executes Python code and returns stdout or stderr."""
    return ""


async def main():
    model_name = "phi4-latest"
    mode: Literal["sequential", "incremental"] = "incremental"
    max_steps = 10
    run_timeout_seconds = 30.0
    
    session_id = random.randint(100000, 999999)
    
    
    # await restart_model(model_name)
    # model = ChatOllama(
    #     model=model_name,
    #     temperature=0,
    #     seed=session_id,
    #     # reasoning=True,
    # )
    
    model = ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )
    
    context_file_names = load_context_files()
    dataset = load_dataset_files()
    
    test_case = dataset[0]
    
    model = model.bind_tools([execute_code])
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context_files="\n".join(context_file_names))},
        {"role": "user", "content": HUMAN_PROMPT.format(question=test_case["question"], guidelines=test_case["guidelines"])},
    ]
    
    session = create_python_async_jitgen_session(
        start_marker="{\"code\":\"", end_marker="\"}", tools={"open": open}
    )
    
    @session.on_stdout
    def on_stdout(stdout: str):
        nonlocal response
        response += stdout

    while True:
        ai_message: AIMessage | None = None
        tool_call_id = None
        response = ""
        async for event in model.astream_events(messages):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                for tool_chunk in chunk.tool_call_chunks:
                    # if name := tool_chunk.get("name"):
                    #     print(f"Tool: {name}")
                    if tool_chunk.get("id"):
                        tool_call_id = tool_chunk["id"]
                    if tool_chunk.get("args"):
                        await session.apush(tool_chunk["args"].encode().decode('unicode_escape'))
                        # print("Tool args chunk:", tool_chunk["args"])

            elif event["event"] == "on_chat_model_end":
                ai_message = event['data']['output']
            else:
                pass
        
        await session.aflush()
        session._algorithm._inside_markers = False
        
        if ai_message.content and not response:
            break
        
        
        messages.append(ai_message)
        messages.append(ToolMessage(content=response, tool_call_id=tool_call_id))
    
    print("Task: ", test_case["question"])
    print("Final response:", ai_message.content)
    print("Correct answer:", test_case["answer"])
    
    
    
    # for test_case in dataset:
    #     agent_session = SequentialAgentSession(
    #         model=model,
    #         context_files=context_file_names,
    #         question=test_case["question"],
    #         guidelines=test_case["guidelines"],
    #         max_steps=max_steps,
    #     ) if mode == "sequential" else IncrementalAgentSession(
    #         model=model,
    #         context_files=context_file_names,
    #         question=test_case["question"],
    #         guidelines=test_case["guidelines"],
    #         max_steps=max_steps,
    #     )
        
    #     try:
    #         result = await asyncio.wait_for(
    #             agent_session.run(), timeout=run_timeout_seconds
    #         )
    #     except asyncio.TimeoutError:
    #         print(
    #             f"Agent run timed out after {run_timeout_seconds:.1f}s for question: "
    #             f"{test_case['question'][:120]}"
    #         )
    #         continue
    
    #     print("Incremental agent success:", result.success)
    #     print("Final result:", result.output)
    #     print("Expected: ", test_case["answer"])
    #     print("Incremental agent total time:", result.total_time)
    #     print("Incremental agent steps executed:", result.steps_executed)
    #     print("Incremental agent steps duration:", ", ".join(f"{d:.2f}s" for d in result.steps_duration))
        
    #     break

if __name__ == "__main__":
    asyncio.run(main())



