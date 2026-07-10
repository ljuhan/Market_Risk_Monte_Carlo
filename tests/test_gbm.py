"""Deterministic tests for GBM simulation and terminal-price metrics."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.risk_engine import gbm as gbm_module
from src.risk_engine.gbm import (
    calculate_empirical_cvar,
    calculate_gbm_var,
    get_simulation_parameters,
    simulate_gbm,
)


def test_get_simulation_parameters_matches_annualized_log_return_formulas() -> None:
    prices = pd.DataFrame(
        {"Close": [100.0, 110.0, 99.0, 108.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    log_returns = np.log(prices["Close"].to_numpy()[1:] / prices["Close"].to_numpy()[:-1])

    s0, mu, sigma = get_simulation_parameters("TEST", data=prices)

    assert s0 == 108.0
    assert mu == pytest.approx(log_returns.mean() * 252)
    assert sigma == pytest.approx(log_returns.std(ddof=1) * np.sqrt(252))


def test_get_simulation_parameters_uses_adjusted_download_settings() -> None:
    downloaded = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    download = Mock(return_value=downloaded)
    original_download = gbm_module.yf.download
    gbm_module.yf.download = download
    try:
        get_simulation_parameters("TEST", start_date="2020-01-01")
    finally:
        gbm_module.yf.download = original_download

    download.assert_called_once_with(
        "TEST", start="2020-01-01", progress=False, auto_adjust=True
    )


def test_simulate_gbm_has_expected_shape_and_initial_row() -> None:
    paths = simulate_gbm(100.0, 0.08, 0.2, 1.0, 5, 7, seed=11)

    assert paths.shape == (6, 7)
    np.testing.assert_allclose(paths.iloc[0].to_numpy(), 100.0)


def test_simulate_gbm_is_reproducible_with_explicit_seed() -> None:
    first = simulate_gbm(100.0, 0.08, 0.2, 1.0, 5, 7, seed=11)
    second = simulate_gbm(100.0, 0.08, 0.2, 1.0, 5, 7, seed=11)

    pd.testing.assert_frame_equal(first, second)


def test_simulate_gbm_zero_volatility_is_deterministic() -> None:
    s0 = 100.0
    mu = 0.10
    horizon = 1.0
    steps = 4
    paths = simulate_gbm(s0, mu, 0.0, horizon, steps, 3, seed=7)
    expected = s0 * np.exp(mu * np.arange(steps + 1) / steps)

    np.testing.assert_allclose(paths.to_numpy(), np.tile(expected[:, None], (1, 3)))


def test_gbm_var_and_cvar_match_independent_terminal_price_formulas() -> None:
    s0 = 100.0
    terminal_prices = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    confidence = 0.8
    threshold = np.percentile(terminal_prices, 20.0)

    expected_var = s0 - threshold
    expected_cvar = s0 - terminal_prices[terminal_prices <= threshold].mean()

    assert calculate_gbm_var(s0, terminal_prices, confidence) == pytest.approx(
        expected_var
    )
    assert calculate_empirical_cvar(s0, terminal_prices, confidence) == pytest.approx(
        expected_cvar
    )


@pytest.mark.parametrize(
    "function",
    [calculate_gbm_var, calculate_empirical_cvar],
)
def test_terminal_price_metrics_reject_invalid_inputs(function) -> None:
    for invalid in [[], [100.0, np.nan], [100.0, 0.0], ["bad", 100.0]]:
        with pytest.raises(ValueError):
            function(100.0, invalid)
    with pytest.raises(ValueError, match="confidence"):
        function(100.0, [90.0, 100.0], confidence=1.0)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.0, 0.1, 0.2, 1.0, 5, 2), "S0"),
        ((100.0, 0.1, -0.1, 1.0, 5, 2), "sigma"),
        ((100.0, 0.1, 0.2, 0.0, 5, 2), "T"),
        ((100.0, 0.1, 0.2, 1.0, 0, 2), "N"),
        ((100.0, 0.1, 0.2, 1.0, 5, 0), "M"),
    ],
)
def test_simulate_gbm_rejects_invalid_parameters(args: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_gbm(*args)


def test_simulate_gbm_rejects_invalid_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        simulate_gbm(100.0, 0.1, 0.2, 1.0, 5, 2, seed="42")


def test_get_simulation_parameters_rejects_empty_nan_and_insufficient_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        get_simulation_parameters("TEST", data=pd.DataFrame())
    with pytest.raises(ValueError, match="finite"):
        get_simulation_parameters(
            "TEST",
            data=pd.DataFrame({"Close": [100.0, np.nan, 102.0]}),
        )
    with pytest.raises(ValueError, match="at least"):
        get_simulation_parameters(
            "TEST",
            data=pd.DataFrame({"Close": [100.0, 101.0]}),
        )
