from pydantic_settings import BaseSettings, SettingsConfigDict

class EvaluationConfig(BaseSettings):
    """
    Configuration settings for the evaluation module.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Timeout for test case execution in seconds
    test_case_timeout: float = 30.0

    # Maximum number of test cases to execute
    max_test_cases: int | None = None

    # Models to be used for evaluation
    models: list[str] = ["llama3.2"]
    
    temperature: float = 0.0
    
    ollama_url: str = "http://host.docker.internal:11434"

    # Directory where results will be saved
    results_directory: str = "./results"
    
    dataset_name: str = "AntonSvystunov/mbpp-jitgen-validation"
    
    dataset: str = "validation"
    
    hf_token: str | None = None
    

config = EvaluationConfig()