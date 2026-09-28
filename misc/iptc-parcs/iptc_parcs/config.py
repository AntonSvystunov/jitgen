from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, get_args

from jitgen_openai import CodeLanguage

from iptc_parcs.providers import PROVIDERS, Provider

LANGUAGES: tuple[CodeLanguage, ...] = get_args(CodeLanguage)
DEFAULT_MODELS = ["lmstudio:qwen/qwen3.6-27b"]


class Strategy(StrEnum):
    BASELINE = "baseline"
    PTC = "ptc"
    IPTC = "iptc"


class Scenario(StrEnum):
    NATURAL = "natural"
    FAULT_COMPILE = "fault_compile"


@dataclass(frozen=True)
class Arm:
    """One way of exposing the tools: a strategy, plus a language for code.

    Attributes:
        strategy: How the model uses the PARCS tools.
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
        max_iterations: Model round trips allowed per agent run.
        executor_timeout: Seconds allowed per executed statement.
        tolerance: Relative error allowed in VaR and CVaR.
        context_length: Context window local models are loaded with.
        tool_result_limit: Maximum characters of each tool message sent to
            the model.
    """

    temperature: float = 0.6
    time_limit: float = 900.0
    max_iterations: int = 12
    executor_timeout: float = 600.0
    tolerance: float = 0.01
    context_length: int = 65536
    tool_result_limit: int = 20000


@dataclass(frozen=True)
class RunSpec:
    """One agent run of the experiment grid."""

    model: ModelSpec
    arm: Arm
    scenario: Scenario
    rep: int
    seed: int
    order: int
    parcs_url: str
    settings: RunSettings = field(default_factory=RunSettings)
    attempt: int = 1

    def key(self) -> dict[str, Any]:
        """The fields identifying this run's cell, as written to the CSVs."""
        return {
            "model": self.model.label,
            "scenario": self.scenario.value,
            "strategy": self.arm.strategy.value,
            "language": self.arm.language,
            "rep": self.rep,
        }
