from dataclasses import dataclass

from mbpp.providers import OLLAMA, Provider


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model to evaluate: which provider serves it, and how to label it."""

    provider: Provider
    model: str  # identifier exactly as the provider's API expects it
    label: str  # human-readable -> ModelLabel column + console output


# Add a model by appending one line here. `provider` is one of the shared
# instances in `mbpp.providers` (or a new `Provider` subclass for a backend
# not covered yet); `model` must match that provider's own identifier
# format exactly (Ollama: "name:tag", LM Studio/OpenRouter: "publisher/name").
MODELS: list[ModelSpec] = [
    # ModelSpec(
    #     OLLAMA, "xingyaow/codeact-agent-mistral:latest", "codeact-mistral (ollama)"
    # ),
    # ModelSpec(OLLAMA, "llama3.2:latest", "llama3.2 (ollama)"),
    # ModelSpec(OLLAMA, "phi4:latest", "Phi-4 (ollama)"),
    # ModelSpec(LMSTUDIO, "google/gemma-4-26b-a4b", "gemma-4-26b-a4b (lmstudio)"),
    # ModelSpec(LMSTUDIO, "openai/gpt-oss-20b", "gpt-oss-20b (lmstudio)"),
    # ModelSpec(LMSTUDIO, "google/gemma-3-4b", "gemma-3-4b (lmstudio)"),
    # ModelSpec(LMSTUDIO, "google/gemma-4-e4b", "gemma-4-e4b (lmstudio)"),
    #
    # Candidates below broaden model/size-class coverage for the generality
    # claim without duplicating an already-tested model. Ollama tags are
    # confirmed against ollama.com/library; LM Studio keys follow this
    # file's existing "publisher/name" pattern but aren't independently
    # confirmed -- check them against a running LM Studio's
    # `/api/v1/models` (or the lmstudio.ai/models search) before use, same
    # as any new entry added here.
    #
    # Dense coder, ~7-8B -- same size class as codeact-mistral (ollama,
    # tested), and a coder/general contrast against llama3.1:8b below.
    # ModelSpec(OLLAMA, "qwen2.5-coder:7b", "qwen2.5-coder-7b (ollama)"),
    # Dense coder, ~14B -- same size class as phi4 (tested).
    # ModelSpec(OLLAMA, "qwen2.5-coder:14b", "qwen2.5-coder-14b (ollama)"),
    # Dense coder, ~22B -- same size class as granite-code:20b below.
    # ModelSpec(OLLAMA, "codestral:22b", "codestral-22b (ollama)"),
    # Dense coder, ~15B -- same size class as phi4 (tested).
    # ModelSpec(OLLAMA, "starcoder2:15b", "starcoder2-15b (ollama)"),
    # Dense coder, ~20B -- same size class as gpt-oss-20b (tested).
    # ModelSpec(OLLAMA, "granite-code:20b", "granite-code-20b (ollama)"),
    # MoE coder, 16B total / ~2.4B active -- same MoE class as gpt-oss-20b
    # and qwen3-coder-30b-a3b-instruct (both tested).
    # ModelSpec(OLLAMA, "deepseek-coder-v2:16b", "deepseek-coder-v2-16b (ollama)"),
    # MoE general, 8x7B total / ~12.9B active -- a larger MoE data point
    # than gpt-oss-20b/qwen3-coder-30b-a3b (both tested).
    # ModelSpec(OLLAMA, "mixtral:8x7b", "mixtral-8x7b (ollama)"),
    # MoE general, 30B total / ~3B active -- the general-purpose sibling of
    # qwen3-coder-30b-a3b-instruct (tested via openrouter below): same
    # architecture and active-param count, coder-tuned vs. general-tuned. qwen3-coder:30b?
    ModelSpec(OLLAMA, "qwen3:30b-a3b", "qwen3-30b-a3b (ollama)"),
    # Dense, ~14B, reasoning-distilled -- same size class as phi4 (tested);
    # instruct-tuned vs. reasoning-tuned contrast at matched size.
    # ModelSpec(OLLAMA, "deepseek-r1:14b", "deepseek-r1-14b (ollama)"),
    # Dense general, ~9B -- mid-size complement to gemma-3-4b (tested).
    # ModelSpec(OLLAMA, "gemma2:9b", "gemma2-9b (ollama)"),
    # Dense general, ~8B -- same size class as qwen2.5-coder:7b above and
    # codeact-mistral (tested); a generation-newer sibling of llama3.2
    # (tested) at the next size up.
    # ModelSpec(OLLAMA, "llama3.1:8b", "llama3.1-8b (ollama)"),
    # Needs OPENROUTER_API_KEY set to run; the orchestration loop skips it
    # with a warning (rather than failing the whole batch) when unset, so
    # this entry is safe to leave registered either way.
    # ModelSpec(OPENROUTER, "openai/gpt-5.6-luna", "gpt-5.6-luna (openrouter)"),
    # ModelSpec(
    #     OPENROUTER,
    #     "qwen/qwen3-coder-30b-a3b-instruct",
    #     "qwen3-coder-30b-a3b-instruct (openrouter)",
    # ),
]

__all__ = ["MODELS", "ModelSpec"]
