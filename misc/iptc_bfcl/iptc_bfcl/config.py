# Adapted from misc/iptc-parcs/iptc_parcs/config.py.
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, get_args

from jitgen_openai import CodeLanguage

from iptc_bfcl.dataset import BfclEntry
from iptc_bfcl.providers import PROVIDERS, Provider

LANGUAGES: tuple[CodeLanguage, ...] = get_args(CodeLanguage)
# OpenRouter's `reasoning.effort` values; "" leaves the model's default.
REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_MODELS = ["lmstudio:qwen/qwen3.6-27b"]


class Strategy(StrEnum):
    BASELINE = "baseline"
    PTC = "ptc"
    IPTC = "iptc"


@dataclass(frozen=True)
class Arm:
    """One way of exposing the tools: a strategy, plus a language for code.

    Attributes:
        strategy: How the model uses the functions.
        language: The `eval` code's language; empty for the baseline, which
            writes no code. Empty rather than `None` so it survives a CSV round
            trip unchanged, which resuming relies on.
    """

    strategy: Strategy
    language: CodeLanguage | str = ""

    @property
    def label(self) -> str:
        return f"{self.strategy}-{self.language}" if self.language else self.strategy


def plan_arms(strategies: list[Strategy], languages: list[CodeLanguage]) -> list[Arm]:
    """Expand strategies into arms: the baseline once, each code strategy per language.

    Args:
        strategies: The strategies to run.
        languages: The languages the code-writing strategies run in.

    Returns:
        The arms, baseline first.
    """
    return [
        arm
        for strategy in strategies
        for arm in (
            [Arm(strategy)]
            if strategy is Strategy.BASELINE
            else [Arm(strategy, language) for language in languages]
        )
    ]


@dataclass(frozen=True)
class ModelSpec:
    """One model: its provider and the provider's model id."""

    provider: Provider
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider.name}:{self.model}"


SINGLE_RESPONSE = "single_response"
FULL_CYCLE = "full_cycle"


def parse_model(spec: str) -> ModelSpec:
    """Parse `provider:model-id`, e.g. `openrouter:deepseek/deepseek-v4.1`.

    Args:
        spec: The model spec.

    Returns:
        The parsed model.

    Raises:
        ValueError: If the provider is unknown or the format is wrong.
    """
    provider_name, sep, model = spec.partition(":")
    if not sep or not model:
        msg = f"model must look like 'provider:model-id', got {spec!r}"
        raise ValueError(msg)
    if provider_name not in PROVIDERS:
        msg = f"unknown provider {provider_name!r}; known: {sorted(PROVIDERS)}"
        raise ValueError(msg)
    return ModelSpec(PROVIDERS[provider_name], model)


@dataclass(frozen=True)
class RunSettings:
    """Everything that must be identical across the arms of one cell.

    Attributes:
        temperature: Sampling temperature for every model call.
        time_limit: Seconds allowed per agent run.
        full_cycle: Run the whole agent loop, feeding tool results back until
            the model answers in text. Off by default, which matches BFCL's
            single-turn scoring: a run ends once its first response, the
            forced tool call, has streamed and its calls have finished.
        max_iterations: Model round trips allowed per agent run with
            `full_cycle`; a run without it makes exactly one.
        executor_timeout: Seconds allowed per executed statement.
        context_length: Context window local models are loaded with.
        tool_result_limit: Maximum characters of each tool message sent to
            the model.
        tool_delay: Seconds every simulated function call takes.
        reasoning_effort: The model's reasoning effort; empty for its default.
        upstream: OpenRouter providers to pin the model to, in order of
            preference, with no fallback to others; empty to let OpenRouter
            route.
        reload_each_run: Reload a local model before every run, not only
            before its first.
    """

    temperature: float = 0.6
    time_limit: float = 180.0
    full_cycle: bool = False
    max_iterations: int = 4
    executor_timeout: float = 30.0
    context_length: int = 65536
    tool_result_limit: int = 20000
    tool_delay: float = 0.1
    reasoning_effort: str = ""
    upstream: tuple[str, ...] = ()
    reload_each_run: bool = False


@dataclass(frozen=True)
class RunSpec:
    """One agent run of the experiment grid."""

    model: ModelSpec
    arm: Arm
    entry: BfclEntry
    rep: int
    seed: int
    order: int
    settings: RunSettings = field(default_factory=RunSettings)
    attempt: int = 1

    def key(self) -> dict[str, Any]:
        """The fields identifying this run's cell, as written to the CSVs."""
        return {
            "model": self.model.label,
            "category": self.entry.category,
            "entry_id": self.entry.id,
            "strategy": self.arm.strategy.value,
            "language": self.arm.language,
            "reasoning_effort": self.settings.reasoning_effort,
            "upstream": ",".join(self.settings.upstream),
            "tool_delay": self.settings.tool_delay,
            "mode": FULL_CYCLE if self.settings.full_cycle else SINGLE_RESPONSE,
            "rep": self.rep,
        }
