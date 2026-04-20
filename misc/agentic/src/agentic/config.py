from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    model_name: str = Field(default="qwen3-coder:30b", alias="MODEL_NAME")
    mode: Literal["incremental", "standard"] = Field(default="incremental", alias="MODE")
    dataset: Literal["dev", "full"] = Field(default="dev", alias="DATASET")
    max_steps: int = Field(default=10, alias="MAX_STEPS")
    seed: int = Field(default=1234, alias="SEED")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    executor: Literal["sandbox", "inproc"] = Field(default="sandbox", alias="EXECUTOR")
    results_dir: str = Field(default="./results", alias="RESULTS_DIR")

    enable_langsmith: bool = Field(default=False, alias="ENABLE_LANGSMITH")
    enable_otel: bool = Field(default=False, alias="ENABLE_OTEL")

    # LM Studio
    lm_studio_host: str = Field(default="localhost:1234", alias="LM_STUDIO_HOST")

    # OpenSandbox (only used when executor=sandbox)
    sandbox_image: str = Field(
        default="agentic-opensandbox-code-interpreter:dabstep", alias="SANDBOX_IMAGE"
    )
    sandbox_domain: str = Field(default="localhost:8080", alias="SANDBOX_DOMAIN")
    sandbox_packages: list[str] = Field(default_factory=lambda: ["pandas", "numpy"], alias="SANDBOX_PACKAGES")

    # Dataset
    dataset_repo_id: str = Field(default="adyen/DABstep", alias="DATASET_REPO_ID")
    context_data_directory: str = Field(default="./tmp/DABstep-data", alias="CONTEXT_DATA_DIRECTORY")
