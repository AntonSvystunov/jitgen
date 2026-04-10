from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .config import AgenticEvaluationConfig

CONTEXT_FILENAMES = [
    "data/context/acquirer_countries.csv",
    "data/context/payments-readme.md",
    "data/context/payments.csv",
    "data/context/merchant_category_codes.csv",
    "data/context/fees.json",
    "data/context/merchant_data.json",
    "data/context/manual.md",
]


@dataclass(slots=True)
class AgenticTask:
    task_id: str
    question: str
    guidelines: str
    answer: str


def load_context_files(config: AgenticEvaluationConfig) -> list[str]:
    resolved_files: list[str] = []

    for filename in CONTEXT_FILENAMES:
        hf_hub_download(
            repo_id=config.dataset_repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=config.context_data_directory,
        )
        resolved_files.append(f"{config.context_data_directory}/{filename}")

    return resolved_files


def load_tasks(config: AgenticEvaluationConfig) -> list[AgenticTask]:
    if config.dataset == "dev":
        dataset = load_dataset(
            config.dataset_repo_id,
            cache_dir=config.dataset_cache_directory,
            download_mode="reuse_dataset_if_exists",
        )["dev"]
    else:
        dataset = load_dataset(
            config.dataset_repo_id,
            name=config.full_dataset_name,
            split=config.full_dataset_split,
            cache_dir=config.dataset_cache_directory,
            download_mode="reuse_dataset_if_exists",
        )

    return [_normalize_task(test_case, index) for index, test_case in enumerate(dataset)]


def _normalize_task(test_case: dict[str, object], index: int) -> AgenticTask:
    return AgenticTask(
        task_id=str(test_case.get("task_id", index)),
        question=str(test_case["question"]),
        guidelines=str(test_case["guidelines"]),
        answer=str(test_case["answer"]),
    )