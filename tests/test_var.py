"""Deterministic tests for VaR calculations."""

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.risk_engine.var import (
    historical_cvar,
    historical_var,
    monte_carlo_normal_var,
    parametric_cvar,
    parametric_var,
)


RETURNS = np.array([-0.08, -0.03, 0.01, 0.02, 0.05], dtype=float)


def test_historical_var_matches_independent_percentile() -> None:
    expected = float(np.percentile(RETURNS, 5.0))

    assert historical_var(RETURNS) == pytest.approx(expected)


def test_historical_cvar_matches_independent_tail_mean() -> None:
    threshold = np.percentile(RETURNS, 5.0)
    expected = float(RETURNS[RETURNS <= threshold].mean())

    assert historical_cvar(RETURNS) == pytest.approx(expected)


def test_parametric_var_matches_independent_normal_formula() -> None:
    alpha = 0.05
    mean = RETURNS.mean()
    sigma = RETURNS.std(ddof=1)
    expected = mean + sigma * NormalDist().inv_cdf(alpha)

    assert parametric_var(RETURNS) == pytest.approx(expected)


def test_parametric_cvar_matches_independent_normal_formula() -> None:
    alpha = 0.05
    mean = RETURNS.mean()
    sigma = RETURNS.std(ddof=1)
    z_score = NormalDist().inv_cdf(alpha)
    density = np.exp(-0.5 * z_score**2) / np.sqrt(2.0 * np.pi)
    expected = mean - sigma * density / alpha

    assert parametric_cvar(RETURNS) == pytest.approx(expected)


def test_monte_carlo_var_is_deterministic_and_matches_independent_draws() -> None:
    confidence = 0.8
    n_simulations = 17
    seed = 123
    rng = np.random.RandomState(seed)
    simulated = rng.normal(RETURNS.mean(), RETURNS.std(ddof=1), n_simulations)
    expected = float(np.percentile(simulated, (1.0 - confidence) * 100.0))

    actual = monte_carlo_normal_var(
        RETURNS,
        confidence=confidence,
        n_simulations=n_simulations,
        seed=seed,
    )

    assert actual == pytest.approx(expected)
    assert actual == monte_carlo_normal_var(
        RETURNS,
        confidence=confidence,
        n_simulations=n_simulations,
        seed=seed,
    )


@pytest.mark.parametrize("function", [historical_var, historical_cvar, parametric_var, parametric_cvar, monte_carlo_normal_var])
def test_var_functions_reject_empty_nan_and_insufficient_inputs(function) -> None:
    for invalid in [np.array([]), np.array([0.01]), np.array([0.01, np.nan])]:
        with pytest.raises(ValueError):
            function(invalid)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_var_functions_reject_invalid_confidence(confidence: float) -> None:
    for function in [historical_var, historical_cvar, parametric_var, parametric_cvar, monte_carlo_normal_var]:
        with pytest.raises(ValueError, match="confidence"):
            function(RETURNS, confidence=confidence)


def test_monte_carlo_rejects_invalid_simulation_count_and_seed() -> None:
    with pytest.raises(ValueError, match="n_simulations"):
        monte_carlo_normal_var(RETURNS, n_simulations=0)
    with pytest.raises(ValueError, match="n_simulations"):
        monte_carlo_normal_var(RETURNS, n_simulations=1.5)
    with pytest.raises(ValueError, match="seed"):
        monte_carlo_normal_var(RETURNS, seed="42")


def test_var_functions_reject_non_numeric_returns() -> None:
    invalid = pd.Series([0.01, "bad"])

    with pytest.raises(ValueError, match="numeric"):
        historical_var(invalid)
