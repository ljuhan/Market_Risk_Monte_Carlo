"""Deterministic tests for return and portfolio-weight calculations."""

import numpy as np
import pandas as pd
import pytest

from src.risk_engine.returns import (
    calculate_portfolio_returns,
    calculate_simple_returns,
    validate_portfolio_weights,
)


def test_calculate_simple_returns_preserves_simple_return_formula() -> None:
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 99.0], "TLT": [50.0, 55.0, 55.0]})

    actual = calculate_simple_returns(prices)

    expected = pd.DataFrame(
        {"SPY": [0.10, -0.10], "TLT": [0.10, 0.0]},
        index=prices.index[1:],
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_validate_portfolio_weights_accepts_matching_unit_weights() -> None:
    validate_portfolio_weights(["SPY", "TLT"], [0.6, 0.4])


@pytest.mark.parametrize(
    ("tickers", "weights", "message"),
    [
        (["SPY"], [0.5, 0.5], "same length"),
        (["SPY", "TLT"], [0.5, 0.4], "sum to 1"),
        ([], [], "tickers cannot be empty"),
        (["SPY", "SPY"], [0.5, 0.5], "duplicates"),
        (["SPY", "TLT"], [np.nan, 1.0], "finite"),
    ],
)
def test_validate_portfolio_weights_rejects_invalid_inputs(
    tickers: list[str], weights: list[float], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_portfolio_weights(tickers, weights)


def test_calculate_portfolio_returns_matches_notebook_weighted_sum() -> None:
    asset_returns = pd.DataFrame(
        {"SPY": [0.10, -0.05], "TLT": [0.02, 0.01]},
        index=["day-1", "day-2"],
    )

    actual = calculate_portfolio_returns(asset_returns, [0.75, 0.25])

    expected = pd.Series([0.08, -0.035], index=asset_returns.index, name=None)
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "asset_returns",
    [
        pd.DataFrame(),
        pd.DataFrame({"SPY": [0.01, np.nan]}),
        pd.DataFrame({"SPY": [0.01, np.inf]}),
        pd.DataFrame({"SPY": [0.01], "TLT": ["bad"]}),
    ],
)
def test_calculate_portfolio_returns_rejects_invalid_data(
    asset_returns: pd.DataFrame,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        calculate_portfolio_returns(asset_returns, [1.0] * max(1, asset_returns.shape[1]))


@pytest.mark.parametrize(
    "prices",
    [
        pd.DataFrame(),
        pd.DataFrame({"SPY": [100.0, np.nan]}),
        pd.DataFrame({"SPY": [100.0, 0.0]}),
        pd.DataFrame({"SPY": [100.0], "TLT": ["bad"]}),
    ],
)
def test_calculate_simple_returns_rejects_invalid_data(
    prices: pd.DataFrame,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        calculate_simple_returns(prices)
