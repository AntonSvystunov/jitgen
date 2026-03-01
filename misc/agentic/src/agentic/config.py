from langchain_ollama import ChatOllama
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgenticEvaluationConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    test_case_timeout: float = 120.0
    max_test_cases: int | None = None
    offset: int = 0
    models: list[str] = ["xingyaow/codeact-agent-mistral"]
    temperature: float = 0.0
    ollama_url: str = "http://host.docker.internal:11434"
    results_directory: str = "./results"
    dataset_name: str = "AntonSvystunov/mbpp-jitgen-validation"
    dataset: str = "validation"
    hf_token: str | None = None
    max_execution_turns: int = 3


config = AgenticEvaluationConfig()


def get_model(model_name: str, session_id: int) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=config.temperature,
        base_url=config.ollama_url,
        seed=session_id,
        keep_alive=0,
        cache=False,
        stop=["</execute>"],
    )
