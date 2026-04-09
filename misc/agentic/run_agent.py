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


async def restart_model(model_name: str):
    await (ChatOllama(
        model=model_name,
        temperature=0,
        reasoning=False,
        keep_alive=0,
        num_predict=2,
    ).ainvoke("Hi"))


async def main():
    model_name = "qwen3-coder:30b"
    mode: Literal["sequential", "incremental"] = "incremental"
    max_steps = 10
    run_timeout_seconds = 30.0
    
    session_id = random.randint(100000, 999999)
    
    
    await restart_model(model_name)
    model = ChatOllama(
        model=model_name,
        temperature=0,
        seed=session_id,
        # reasoning=True,
    )
    
    context_file_names = load_context_files()
    dataset = load_dataset_files()
    
    
    for test_case in dataset:
        agent_session = SequentialAgentSession(
            model=model,
            context_files=context_file_names,
            question=test_case["question"],
            guidelines=test_case["guidelines"],
            max_steps=max_steps,
        ) if mode == "sequential" else IncrementalAgentSession(
            model=model,
            context_files=context_file_names,
            question=test_case["question"],
            guidelines=test_case["guidelines"],
            max_steps=max_steps,
        )
        
        try:
            result = await asyncio.wait_for(
                agent_session.run(), timeout=run_timeout_seconds
            )
        except asyncio.TimeoutError:
            print(
                f"Agent run timed out after {run_timeout_seconds:.1f}s for question: "
                f"{test_case['question'][:120]}"
            )
            continue
    
        print("Incremental agent success:", result.success)
        print("Final result:", result.output)
        print("Expected: ", test_case["answer"])
        print("Incremental agent total time:", result.total_time)
        print("Incremental agent steps executed:", result.steps_executed)
        print("Incremental agent steps duration:", ", ".join(f"{d:.2f}s" for d in result.steps_duration))

if __name__ == "__main__":
    asyncio.run(main())



