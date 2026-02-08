from langchain_ollama import ChatOllama
from datasets import load_dataset
from .chains import create_jitgen_chain, create_sync_executor_chain
from .utils import run_test_cases, TqdmLoggingHandler
from random import randint
import logging
import sys
from tqdm import tqdm

from .config import config

# Configure logging with tqdm-compatible handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler for detailed logs
file_handler = logging.FileHandler("jitgen_eval.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Tqdm-compatible console handler
tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(tqdm_handler)

MODEL_NAME = "llama3.2"
MODE = "async"
DATASET = "validation"
OUTPUT_FILE = f"./results/{MODEL_NAME.replace(':', '_').replace('.', '_')}_{MODE}_chain_results_{DATASET}.csv"


async def run_evaluation():
    session_id = randint(0, 1000000)
    tqdm.write(f"\n{'=' * 60}", file=sys.stderr)
    tqdm.write("🚀 Starting JITGEN Evaluation", file=sys.stderr)
    tqdm.write(f"   Session ID: {session_id}", file=sys.stderr)
    tqdm.write(f"{'=' * 60}\n", file=sys.stderr)

    tqdm.write("📦 Loading dataset...", file=sys.stderr)
    dataset_full = load_dataset(
        config.dataset_name, cache_dir="./data", download_mode="reuse_dataset_if_exists"
    )
    tqdm.write(f"✅ Dataset loaded: {config.dataset_name}\n", file=sys.stderr)

    for model_name in config.models:
        tqdm.write(f"\n{'=' * 60}", file=sys.stderr)
        tqdm.write(f"🤖 Evaluating model: {model_name}", file=sys.stderr)
        tqdm.write(f"{'=' * 60}\n", file=sys.stderr)
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
            dataset_full[config.dataset].select(
                range(
                    config.offset,
                    config.offset
                    + (config.max_test_cases or len(dataset_full[config.dataset])),
                )
            )
            if config.max_test_cases
            else dataset_full[config.dataset].select(
                range(config.offset, len(dataset_full[config.dataset]))
            )
        )
        tqdm.write(
            f"📊 Dataset: {config.dataset} ({len(target_dataset)} test cases)\n",
            file=sys.stderr,
        )

        tqdm.write("🔄 Running ASYNC chain evaluation...", file=sys.stderr)
        async_chain_results = await run_test_cases(llm, target_dataset, jitgen_chain)
        output_file = f"{config.results_directory}/{model_name.replace(':', '_').replace('.', '_')}_async_chain_results_{config.dataset}.csv"
        async_chain_results.to_csv(output_file, index=False)
        tqdm.write(f"💾 Async results saved: {output_file}\n", file=sys.stderr)

        tqdm.write("🔄 Running SYNC chain evaluation...", file=sys.stderr)
        sync_chain_results = await run_test_cases(llm, target_dataset, sync_chain)
        output_file = f"{config.results_directory}/{model_name.replace(':', '_').replace('.', '_')}_sync_chain_results_{config.dataset}.csv"
        sync_chain_results.to_csv(output_file, index=False)
        tqdm.write(f"💾 Sync results saved: {output_file}\n", file=sys.stderr)

    tqdm.write(f"\n{'=' * 60}", file=sys.stderr)
    tqdm.write("🎉 Evaluation completed successfully!", file=sys.stderr)
    tqdm.write(f"{'=' * 60}\n", file=sys.stderr)
