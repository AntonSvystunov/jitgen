import numpy as np
import pytest

from iptc_parcs.task import (
    N_ASSETS,
    RHO,
    extract_answer,
    reference,
    relative_error,
)


def test_reference_matches_an_independent_monte_carlo():
    s = np.array([0.01 + 0.001 * i for i in range(N_ASSETS)])
    lags = np.abs(np.subtract.outer(np.arange(N_ASSETS), np.arange(N_ASSETS)))
    cholesky = np.linalg.cholesky(np.outer(s, s) * RHO**lags)
    rng = np.random.default_rng(7)
    losses = -(rng.standard_normal((1_000_000, N_ASSETS)) @ cholesky.T).mean(axis=1)
    var = np.quantile(losses, 0.99)
    cvar = losses[losses >= var].mean()

    ref = reference()
    assert relative_error(var, ref.var_99) < 0.005
    assert relative_error(cvar, ref.cvar_99) < 0.005
    assert ref.cvar_99 / ref.var_99 == pytest.approx(1.1455, abs=1e-3)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('Done.\n{"var_99": 0.0176, "cvar_99": 0.0201}', (0.0176, 0.0201)),
        ('```json\n{"var_99": "0.0176", "cvar_99": 0.0201}\n```', (0.0176, 0.0201)),
        (
            'first {"var_99": 1, "cvar_99": 2} then {"var_99": 3, "cvar_99": 4}',
            (3.0, 4.0),
        ),
        ('{"var": 0.1} and no answer', None),
        ("no json at all", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"VaR": 0.0176, "CVaR": 0.0201, "N": 2000000}', (0.0176, 0.0201)),
        ('{"var99": 0.0176, "cvar99": 0.0201}', (0.0176, 0.0201)),
        ('{"VaR_99": 0.0176, "ES_99": 0.0201}', (0.0176, 0.0201)),
        ('{"var_99": "n/a", "cvar_99": 0.0201}', None),
        ('{"VaR": 0.0176}', None),
    ],
)
def test_extract_answer_accepts_aliased_keys(text, expected):
    assert extract_answer(text) == expected
