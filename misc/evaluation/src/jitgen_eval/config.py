from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationConfig(BaseSettings):
    """
    Configuration settings for the evaluation module.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Timeout for test case execution in seconds
    test_case_timeout: float = 30.0

    # Maximum number of test cases to execute
    max_test_cases: int | None = None

    # Offset for selecting test cases from the dataset
    offset: int = 0

    # Models to be used for evaluation
    models: list[str] = ["llama3.2"]

    temperature: float = 0.0

    ollama_url: str = "http://host.docker.internal:11434"

    # Directory where results will be saved
    results_directory: str = "./results"

    dataset_name: str = "AntonSvystunov/mbpp-jitgen-validation"

    dataset: str = "validation"

    hf_token: str | None = None

    openrouter_api_key: str | None = None


config = EvaluationConfig()

def get_model(model_name: str, session_id: str) -> ChatOpenRouter:
    if "/" in model_name:
        return ChatOpenRouter(
            model_name=model_name,
            temperature=0,
            seed=session_id,
            cache=False,
            api_key=config.openrouter_api_key,
        )

    return ChatOllama(
        model=model_name,
        temperature=0,
        base_url=config.ollama_url,
        seed=session_id,
        keep_alive=0,
        cache=False,
    )

