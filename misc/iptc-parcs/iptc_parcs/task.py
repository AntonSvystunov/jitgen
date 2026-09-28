import json
import math
import re
from dataclasses import dataclass
from statistics import NormalDist

N_ASSETS = 20
RHO = 0.5
CONFIDENCE = 0.99
SCENARIOS = 2_000_000

TASK = (
    "Using the PARCS cluster, estimate the 1-day 99% Value-at-Risk (VaR) and "
    "Conditional VaR (CVaR, expected shortfall) of an equally weighted portfolio "
    f"of {N_ASSETS} assets (weight 1/{N_ASSETS} each). Daily asset returns are "
    "jointly normal with zero mean and covariance "
    f"Sigma[i][j] = s[i] * s[j] * {RHO}^|i - j|, where s[i] = 0.01 + 0.001 * i "
    f"for i = 0..{N_ASSETS - 1}. Loss = -(portfolio return). Use Monte Carlo "
    f"with {SCENARIOS:,} scenarios in total, split evenly across the workers of "
    "a first layer: each worker draws its own scenarios with its own seed, using "
    "the Cholesky factor of Sigma. Then compute the 99% quantile of all losses "
    "(VaR) and the mean of the losses at or beyond it (CVaR) in a single-worker "
    "aggregation layer."
)


@dataclass(frozen=True)
class Reference:
    """Exact 99% VaR and CVaR of the task's portfolio loss."""

    var_99: float
    cvar_99: float


def reference() -> Reference:
    """Compute the task's exact answer.

    The loss is normal with zero mean and standard deviation
    sqrt(w' Sigma w), so VaR = z * sigma and CVaR = sigma * pdf(z) / (1 - p).

    Returns:
        The exact VaR and CVaR.
    """
    s = [0.01 + 0.001 * i for i in range(N_ASSETS)]
    w = 1.0 / N_ASSETS
    variance = sum(
        w * w * s[i] * s[j] * RHO ** abs(i - j)
        for i in range(N_ASSETS)
        for j in range(N_ASSETS)
    )
    sigma = math.sqrt(variance)
    normal = NormalDist()
    z = normal.inv_cdf(CONFIDENCE)
    return Reference(z * sigma, sigma * normal.pdf(z) / (1 - CONFIDENCE))


_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def extract_answer(text: str) -> tuple[float, float] | None:
    """Find the last JSON object in `text` holding `var_99` and `cvar_99`.

    Args:
        text: The model's final answer.

    Returns:
        `(var_99, cvar_99)`, or `None` if no such object is present.
    """
    for candidate in reversed(_JSON_OBJECT.findall(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            return float(data["var_99"]), float(data["cvar_99"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def relative_error(value: float, exact: float) -> float:
    """Relative error of `value` against `exact`."""
    return abs(value - exact) / abs(exact)
