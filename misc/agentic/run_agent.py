import asyncio
from dataclasses import dataclass
import random
import re

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from jitgen.executors.python import InProcPythonExecutor
from jitgen.prebuilt.python import create_python_async_jitgen_session
from langchain_ollama import ChatOllama
from datasets import load_dataset

from time import perf_counter

from langchain_core.language_models import BaseChatModel

from langsmith import traceable

load_dotenv(override=True)


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


SYSTEM_PROMPT = """
You are a helpful assistant assigned with the task of problem-solving. To achieve this, \
you will be using an interactive coding environment equipped with a variety of tool \
functions to assist you throughout the process.

After that, you have two options:
1) Interact with a Python programming environment and receive the corresponding output.
Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>.
Note that your environment persists across interactions, so you can define variables and functions that can be used in subsequent code executions.
2) Directly provide a solution that adheres to the required format for the given task.
Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>.

## Exploring the environment:
Use Python to read "*.md" files to understand the data and then read "*.csv" and "*.json" files to explore the data.
Use <execute> block to read .md file and print the content. On next observation, you will be provided with stdout of code block execution.

For example, to read contents of a .md file, you can write:
<execute>
with open("<path-to-md-file>", "r") as f:
    content = f.read()
print(content)
</execute>

IMPORTANT! Do not print content of .csv or .json files directly as it can be very large. Instead, use Python to explore the data and print only relevant information.
## Available files:
You have these files available:
{context_files}

Note: *.md files contain documentation about the data, while *.csv and *.json files contain the actual data.
""".strip()

HUMAN_PROMPT = """
Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""
question = "What are the unique set of merchants in the payments data?"
guidelines = "Answer with a comma separated list"


@traceable
async def execute_incremental_agent(model: BaseChatModel, system_prompt: str, question: str, guidelines: str) -> str | None:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": HUMAN_PROMPT.format(
            question=question,
            guidelines=guidelines,
        )},
    ]
    
    steps_left = 5
    
    start_time = perf_counter()
    async with create_python_async_jitgen_session(
        start_marker="<execute>", end_marker="</execute>", tools={"open": open}
    ) as session:
        observation = "Observation:\n"
        execute_called = False
        
        @session.on_stdout
        def on_stdout(stdout: str):
            nonlocal observation
            nonlocal execute_called
            execute_called = True
            observation += stdout
        
        @session.on_error
        def on_error(stderr: str):
            nonlocal observation
            nonlocal execute_called
            execute_called = True
            observation += stderr
    
        while steps_left > 0:
            observation = ""
            execute_called = False
            response = ""

            async for chunk in model.astream(messages):
                chunk_text = chunk.content
                response += chunk_text
                await session.apush(chunk_text)

            await session.aflush()
            # print("Response:")
            # print(response)
            
            # print("Observation:")
            # print(observation)
            
            if "<solution>" in response:
                # print("Solution proposed, stopping interaction.")
                break
            
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Observation:\n" + observation})
            steps_left -= 1
    
    end_time = perf_counter()

    solution_blocks = re.findall(r"<solution>(.*?)</solution>", response, re.DOTALL)
    
    final_response = solution_blocks[0].strip() if solution_blocks else None
    
    return ExecutionResult(
        success=bool(final_response),
        output=final_response,
        total_time=end_time - start_time,
        steps_executed=5 - steps_left,
    )

# async def _execute_python_code(text: str) -> str:
#     code_block = re.search(r"<execute>(.+)", text, re.DOTALL)
#     if not code_block:
#         return ""

#     source_code = code_block.group(1)

#     executor = InProcPythonExecutor()
#     result = await executor.aexecute(source_code)

#     if result.success:
#         return result.output
#     else:
#         raise ValueError(f"Error detected. Halting further processing. {result.error}")


@dataclass
class ExecutionResult:
    success: bool
    output: str | None
    total_time: float
    steps_executed: int


@traceable
async def execute_sequential_agent(model: BaseChatModel, system_prompt: str, question: str, guidelines: str) -> ExecutionResult:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": HUMAN_PROMPT.format(
            question=question,
            guidelines=guidelines,
        )},
    ]
    
    steps_left = 5
    
    executor = InProcPythonExecutor()
    start_time = perf_counter()
    while steps_left > 0:
        response = ""
        async for chunk in model.astream(messages):
            chunk_text = chunk.content
            response += chunk_text
        
            # print("Response:")
            # print(response)
        
        if "<solution>" in response:
            # print("Solution proposed, stopping interaction.")
            break
        
        if "<execute>" in response:
            code_block = re.search(r"<execute>(.+)</execute>", response, re.DOTALL)
            if code_block:
                source_code = code_block.group(1)
                # print("Executing code block:")
                # print(source_code)
                result = await executor.aexecute(source_code)

                if result.success:
                    observation = result.output or ""
                    # print("Execution output:")
                    # print(observation)
                else:
                    observation = result.error or "Error during code execution."
                    # print("Execution error:")
                    # print(observation)

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Observation:\n" + observation})
        
        steps_left -= 1
    
    end_time = perf_counter()

    solution_blocks = re.findall(r"<solution>(.*?)</solution>", response, re.DOTALL)
    
    final_response = solution_blocks[0].strip() if solution_blocks else None
    
    return ExecutionResult(
        success=bool(final_response),
        output=final_response,
        total_time=end_time - start_time,
        steps_executed=5 - steps_left,
    )


async def main():
    session_id = random.randint(100000, 999999)
    
    model = ChatOllama(
        model="gemma4:latest",
        temperature=0,
        seed=session_id,
        reasoning=False,
    )
    
    context_file_names = load_context_files()
    dataset = load_dataset_files()
    
    test_case = dataset[9]
    
    system_prompt = SYSTEM_PROMPT.format(context_files="\n".join(context_file_names))
    
    sequential_result = await execute_sequential_agent(model, system_prompt, test_case["question"], test_case["guidelines"])
    if sequential_result.success:
        print("Final result:", sequential_result.output)
        print("Expected: ", test_case["answer"])
        print("Sequential agent total time:", sequential_result.total_time)

    print("\n\n====================\n\n")
    
    incremental_result = await execute_incremental_agent(model, system_prompt, test_case["question"], test_case["guidelines"])
    if incremental_result.success:
        print("Final result:", incremental_result.output)
        print("Expected: ", test_case["answer"])
        print("Incremental agent total time:", incremental_result.total_time)


if __name__ == "__main__":
    asyncio.run(main())



