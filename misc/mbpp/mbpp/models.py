from dataclasses import dataclass

from mbpp.providers import LMSTUDIO, OLLAMA, Provider


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model to evaluate: which provider serves it, and how to label it."""

    provider: Provider
    model: str  # identifier exactly as the provider's API expects it
    label: str  # human-readable -> ModelLabel column + console output


# Add a model by appending one line here. `provider` is one of the shared
# instances in `mbpp.providers` (OLLAMA, LMSTUDIO, OPENROUTER) -- import
# whichever ones are actually referenced by an uncommented entry, since
# every entry below is currently commented out; `model` must match that
# provider's own identifier format exactly (Ollama: "name:tag", LM
# Studio/OpenRouter: "publisher/name").
#
# Every entry below already has a matching pair of CSVs checked in under
# `results/`. `label` is normalized to `model` with its provider/org prefix
# (the part up to and including the last `/`) stripped and `:` replaced
# with `-` -- the same shape `sanitize_filename_component` (`results.py`)
# gives the CSV stem, so a result file can always be traced back to its
# `ModelSpec` by eye.
MODELS: list[ModelSpec] = [
    # ModelSpec(
    #     OLLAMA,
    #     "xingyaow/codeact-agent-mistral:latest",
    #     "codeact-agent-mistral-latest (ollama)",
    # ),
    # ModelSpec(OLLAMA, "llama3.2:latest", "llama3.2-latest (ollama)"),
    # ModelSpec(OLLAMA, "phi4:latest", "phi4-latest (ollama)"),
    ModelSpec(LMSTUDIO, "google/gemma-3-4b", "gemma-3-4b (lmstudio)"),
    ModelSpec(LMSTUDIO, "google/gemma-4-e4b", "gemma-4-e4b (lmstudio)"),
    ModelSpec(LMSTUDIO, "openai/gpt-oss-20b", "gpt-oss-20b (lmstudio)"),
    ModelSpec(OLLAMA, "qwen3:30b-a3b", "qwen3-30b-a3b (ollama)"),
    ModelSpec(OLLAMA, "mistral-nemo:12b", "mistral-nemo-12b (ollama)"),
    ModelSpec(OLLAMA, "smollm2:1.7b", "smollm2-1.7b (ollama)"),
    ModelSpec(OLLAMA, "zephyr:7b", "zephyr-7b (ollama)"),
]

__all__ = ["MODELS", "ModelSpec"]
