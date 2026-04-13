from pathlib import Path

from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset

_DATASET_ID = "adyen/DABstep"

_CONTEXT_FILENAMES = [
    "data/context/acquirer_countries.csv",
    "data/context/payments-readme.md",
    "data/context/payments.csv",
    "data/context/merchant_category_codes.csv",
    "data/context/fees.json",
    "data/context/merchant_data.json",
    "data/context/manual.md",
]


def load_context_files(base_dir: str) -> list[str]:
    for filename in _CONTEXT_FILENAMES:
        hf_hub_download(
            repo_id=_DATASET_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=base_dir,
            # force_download=True
        )

    return [str(Path(base_dir) / filename) for filename in _CONTEXT_FILENAMES]


def load_tasks_dataset(split: str) -> Dataset:
    dataset = load_dataset(_DATASET_ID, name="tasks", split=split)

    return dataset
