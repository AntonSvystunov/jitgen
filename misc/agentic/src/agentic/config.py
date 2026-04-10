from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic_settings import BaseSettings, SettingsConfigDict

AgentMode = Literal["sequential", "incremental"]
DatasetSelection = Literal["dev", "full"]


class AgenticEvaluationConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = "qwen3-coder:30b"
    mode: AgentMode = "incremental"
    dataset: DatasetSelection = "dev"
    max_steps: int = 10
    max_timeout: float = 30.0
    session_id: int = 623444

    ollama_url: str = "http://localhost:11434"
    results_directory: str = "./results"
    dataset_repo_id: str = "adyen/DABstep"
    full_dataset_name: str = "tasks"
    full_dataset_split: str = "default"
    dataset_cache_directory: str = "./data"
    context_data_directory: str = "./tmp/DABstep-data"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run agentic evaluation against the DABstep dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", dest="model_name", default=None, help="Model name")
    parser.add_argument(
        "--model-name",
        dest="model_name",
        default=None,
        help="Model name",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "incremental"],
        default=None,
        help="Agent execution mode",
    )
    parser.add_argument(
        "--dataset",
        choices=["dev", "full"],
        default=None,
        help="Dataset selection",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per task",
    )
    parser.add_argument(
        "--max-timeout",
        type=float,
        default=None,
        help="Maximum timeout per task in seconds",
    )
    return parser


def load_config(argv: Sequence[str] | None = None) -> AgenticEvaluationConfig:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    overrides = {
        key: value for key, value in vars(args).items() if value is not None
    }
    return AgenticEvaluationConfig(**overrides)


def create_model(config: AgenticEvaluationConfig) -> ChatOllama:
    return ChatOllama(
        model=config.model_name,
        temperature=0,
        base_url=config.ollama_url,
        seed=config.session_id,
        # cache=False,
        # keep_alive=0,
        reasoning=False,
    )
    
async def reinit_model(config: AgenticEvaluationConfig):
    await (ChatOllama(
        model=config.model_name,
        temperature=0,
        base_url=config.ollama_url,
        cache=False,
        keep_alive=0,
        reasoning=False,
        num_predict=10,
    ).ainvoke("Hi"))