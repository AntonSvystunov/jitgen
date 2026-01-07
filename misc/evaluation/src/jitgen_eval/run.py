from langchain_ollama import ChatOllama
from datasets import load_dataset
from .chains import create_jitgen_chain, create_sync_executor_chain
from .utils import run_test_cases
from random import randint

from .config import config

MODEL_NAME = "llama3.2"
MODE = "async"
DATASET = "validation"
OUTPUT_FILE = f"./results/{MODEL_NAME.replace(':', '_').replace('.', '_')}_{MODE}_chain_results_{DATASET}.csv"

async def run_evaluation():
    session_id = randint(0, 1000000)
    
    dataset_full = load_dataset(
        config.dataset_name, cache_dir="./data", download_mode="reuse_dataset_if_exists"
    )
    for model_name in config.models:
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=config.ollama_url,
            seed=session_id,
            keep_alive=0,
            cache=False,
        )

        jitgen_chain = create_jitgen_chain(llm)
        sync_chain = create_sync_executor_chain(llm)

        target_dataset = (
            dataset_full[config.dataset].select(range(config.max_test_cases))
            if config.max_test_cases
            else dataset_full[config.dataset]
        )

        async_chain_results = await run_test_cases(llm, target_dataset, jitgen_chain)
        async_chain_results.to_csv(
            f"{config.results_directory}/{model_name.replace(':', '_').replace('.', '_')}_async_chain_results_{config.dataset}.csv",
            index=False,
        )

        sync_chain_results = await run_test_cases(llm, target_dataset, sync_chain)
        sync_chain_results.to_csv(
            f"{config.results_directory}/{model_name.replace(':', '_').replace('.', '_')}_sync_chain_results_{config.dataset}.csv",
            index=False,
        )
