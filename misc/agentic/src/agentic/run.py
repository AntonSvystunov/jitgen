import logging
import sys
from random import randint

from datasets import load_dataset
from tqdm import tqdm

from .config import config, get_model
from .graph import create_agent_graph
from .utils import get_results_file_name, run_test_cases, TqdmLoggingHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("agentic_eval.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(tqdm_handler)


async def run_evaluation():
    session_id = randint(0, 1_000_000)
    tqdm.write(f"\n{'=' * 60}", file=sys.stderr)
    tqdm.write("🚀 Starting CodeAct Agentic Evaluation", file=sys.stderr)
    tqdm.write(f"   Session ID: {session_id}", file=sys.stderr)
    tqdm.write(f"   Max execution turns: {config.max_execution_turns}", file=sys.stderr)
    tqdm.write(f"{'=' * 60}\n", file=sys.stderr)

    tqdm.write("📦 Loading dataset...", file=sys.stderr)
    dataset_full = load_dataset(
        config.dataset_name,
        cache_dir="./data",
        download_mode="reuse_dataset_if_exists",
    )
    tqdm.write(f"✅ Dataset loaded: {config.dataset_name}\n", file=sys.stderr)

    for model_name in config.models:
        tqdm.write(f"\n{'=' * 60}", file=sys.stderr)
        tqdm.write(f"🤖 Evaluating model: {model_name}", file=sys.stderr)
        tqdm.write(f"{'=' * 60}\n", file=sys.stderr)
        llm = get_model(model_name, session_id)

        # Build graphs for both strategies
        async_graph = create_agent_graph(llm, "async", config)
        sync_graph = create_agent_graph(llm, "sync", config)

        # Select dataset slice
        full_ds = dataset_full[config.dataset]
        end_idx = config.offset + (config.max_test_cases or len(full_ds))
        end_idx = min(end_idx, len(full_ds))
        target_dataset = full_ds.select(range(config.offset, end_idx))

        tqdm.write(
            f"📊 Dataset: {config.dataset} ({len(target_dataset)} test cases)\n",
            file=sys.stderr,
        )

        # --- Async (JITGen) agent evaluation ---
        tqdm.write("🔄 Running ASYNC (JITGen) agent evaluation...", file=sys.stderr)
        async_results = await run_test_cases(
            async_graph, target_dataset, config, "async_agent"
        )
        output_file = get_results_file_name(
            config.results_directory, model_name, config.dataset, "async_agent"
        )
        async_results.to_csv(output_file, index=False)
        tqdm.write(f"💾 Async agent results saved: {output_file}\n", file=sys.stderr)

        # --- Sync agent evaluation ---
        tqdm.write("🔄 Running SYNC agent evaluation...", file=sys.stderr)
        sync_results = await run_test_cases(
            sync_graph, target_dataset, config, "sync_agent"
        )
        output_file = get_results_file_name(
            config.results_directory, model_name, config.dataset, "sync_agent"
        )
        sync_results.to_csv(output_file, index=False)
        tqdm.write(f"💾 Sync agent results saved: {output_file}\n", file=sys.stderr)

    tqdm.write(f"\n{'=' * 60}", file=sys.stderr)
    tqdm.write("🎉 Agentic evaluation completed successfully!", file=sys.stderr)
    tqdm.write(f"{'=' * 60}\n", file=sys.stderr)
