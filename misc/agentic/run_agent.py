import asyncio
from dataclasses import dataclass
import random
import re

from agentic.agents import IncrementalAgentSession, IncrementalAgentSession, SequentialAgentSession
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


async def main():
    session_id = random.randint(100000, 999999)
    
    model = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        seed=session_id,
        reasoning=False,
    )
    
    context_file_names = load_context_files()
    dataset = load_dataset_files()
    
    test_case = dataset[1]
    
    
    sequestial_session = SequentialAgentSession(
        model=model,
        context_files=context_file_names,
        question=test_case["question"],
        guidelines=test_case["guidelines"],
        max_steps=5,
    )
    
    
    sequential_result = await sequestial_session.run()
    print("Sequential agent success:", sequential_result.success)
    print("Final result:", sequential_result.output)
    print("Expected: ", test_case["answer"])
    print("Sequential agent total time:", sequential_result.total_time)
    print("Sequential agent steps executed:", sequential_result.steps_executed)
    print("Sequential agent steps duration:", ", ".join(f"{d:.2f}s" for d in sequential_result.steps_duration))

    print("\n\n====================\n\n")
    
    incremental_agent = IncrementalAgentSession(
        model=model,
        context_files=context_file_names,
        question=test_case["question"],
        guidelines=test_case["guidelines"],
        max_steps=5,
    )
    
    incremental_result = await incremental_agent.run()
    
    print("Incremental agent success:", incremental_result.success)
    print("Final result:", incremental_result.output)
    print("Expected: ", test_case["answer"])
    print("Incremental agent total time:", incremental_result.total_time)
    print("Incremental agent steps executed:", incremental_result.steps_executed)
    print("Incremental agent steps duration:", ", ".join(f"{d:.2f}s" for d in incremental_result.steps_duration))

if __name__ == "__main__":
    asyncio.run(main())



