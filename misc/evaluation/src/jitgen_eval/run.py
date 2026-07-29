from datasets import load_dataset
from .chains import create_jitgen_chain, create_sync_executor_chain
from .utils import get_results_file_name, run_test_cases, TqdmLoggingHandler
from functools import partial
from random import randint
import logging
import sys
from tqdm import tqdm

from .config import config, get_model

# Configure logging with tqdm-compatible handler.  Attach to the *package*
# logger, not __name__ — otherwise records from sibling modules such as
# jitgen_eval.utils propagate straight past these handlers to the root logger
# and never reach the log file.
package_logger = logging.getLogger(__package__)
package_logger.setLevel(logging.INFO)

# File handler for detailed logs
file_handler = logging.FileHandler("jitgen_eval.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
package_logger.addHandler(file_handler)

# Tqdm-compatible console handler
tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
package_logger.addHandler(tqdm_handler)

logger = logging.getLogger(__name__)


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
        llm = get_model(model_name, session_id)

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

        # One pass per arm, each preceded by an identical warm-up.  See
        # run_test_cases: this is what makes both arms generate the *same*
        # program, which matters more than the ~1% tailwind that interleaving
        # would have balanced.
        tqdm.write("🔄 Running ASYNC/SYNC chain evaluation...", file=sys.stderr)
        results = await run_test_cases(
            llm,
            target_dataset,
            {
                "async_chain": partial(create_jitgen_chain, llm),
                "sync_chain": partial(create_sync_executor_chain, llm),
            },
        )

        for chain_type, df in results.items():
            output_file = get_results_file_name(
                config.results_directory, model_name, config.dataset, chain_type
            )
            df.to_csv(output_file, index=False)
            tqdm.write(f"💾 {chain_type} results saved: {output_file}\n", file=sys.stderr)

    tqdm.write(f"\n{'=' * 60}", file=sys.stderr)
    tqdm.write("🎉 Evaluation completed successfully!", file=sys.stderr)
    tqdm.write(f"{'=' * 60}\n", file=sys.stderr)
