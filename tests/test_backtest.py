"""Deterministic tests for rolling historical VaR backtesting."""

import numpy as np
import pandas as pd
import pytest

from src.risk_engine.backtest import (
    backtest_var,
    classify_backtest_exceptions,
    rolling_historical_var,
)


def test_rolling_historical_var_matches_prior_window_percentiles() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04, 0.05],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    window = 3

    actual = rolling_historical_var(returns, window=window)
    expected = pd.Series(
        [np.percentile(returns.iloc[0:3], 5.0), np.percentile(returns.iloc[1:4], 5.0)],
        index=pd.DatetimeIndex(returns.index[3:].tolist()),
    )

    pd.testing.assert_series_equal(actual, expected)


def test_rolling_historical_var_excludes_current_observation() -> None:
    original = pd.Series(
        [0.01, -0.02, 0.03, -0.04, 0.05],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    changed_current = original.copy()
    changed_current.iloc[3] = -0.90

    original_var = rolling_historical_var(original, window=3)
    changed_var = rolling_historical_var(changed_current, window=3)

    assert original_var.iloc[0] == changed_var.iloc[0]
    assert original_var.iloc[0] == pytest.approx(
        np.percentile(original.iloc[:3], 5.0)
    )


def test_backtest_var_uses_strict_less_than_exception_rule() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    actual = pd.Series([-0.05, -0.02, -0.03], index=index)
    predicted = pd.Series([-0.05, -0.02, -0.01], index=index)

    result = backtest_var(actual, predicted)

    expected_exceptions = pd.Series([False, False, True], index=index)
    pd.testing.assert_series_equal(result["exceptions"], expected_exceptions)
    assert result["exception_count"] == 1
    assert result["observation_count"] == 3
    assert result["exception_rate"] == pytest.approx(1 / 3)


def test_backtest_var_returns_structured_outputs_and_classification() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    actual = pd.Series([-0.10, -0.20, 0.01, -0.30], index=index)
    predicted = pd.Series([-0.05, -0.05, -0.05, -0.05], index=index)

    result = backtest_var(actual, predicted)

    assert set(result) == {
        "predicted_var",
        "exceptions",
        "exception_count",
        "observation_count",
        "exception_rate",
        "classification",
    }
    pd.testing.assert_series_equal(result["predicted_var"], predicted)
    assert result["exception_count"] == 3
    assert result["observation_count"] == 4
    assert result["exception_rate"] == pytest.approx(0.75)
    assert result["classification"] == "red"


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.05, "acceptable"),
        (0.050001, "yellow"),
        (0.08, "yellow"),
        (0.080001, "red"),
    ],
)
def test_classify_backtest_exceptions_preserves_notebook_thresholds(
    rate: float, expected: str
) -> None:
    assert classify_backtest_exceptions(rate) == expected


@pytest.mark.parametrize("returns", [pd.Series(dtype=float), pd.Series([0.01])])
def test_rolling_historical_var_rejects_empty_or_insufficient_returns(
    returns: pd.Series,
) -> None:
    with pytest.raises(ValueError):
        rolling_historical_var(returns, window=1)


def test_rolling_historical_var_rejects_nan_and_invalid_window_or_confidence() -> None:
    returns = pd.Series([0.01, np.nan, 0.03], index=pd.RangeIndex(3))

    with pytest.raises(ValueError, match="finite"):
        rolling_historical_var(returns, window=1)
    valid_returns = returns.fillna(0.02)
    with pytest.raises(ValueError, match="window"):
        rolling_historical_var(valid_returns, window=0)
    with pytest.raises(ValueError, match="confidence"):
        rolling_historical_var(valid_returns, window=1, confidence=1.0)


def test_backtest_var_rejects_empty_nan_and_mismatched_indices() -> None:
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    actual = pd.Series([0.01, np.nan], index=index)
    predicted = pd.Series([0.0, 0.0], index=index)

    with pytest.raises(ValueError, match="finite"):
        backtest_var(actual, predicted)
    with pytest.raises(ValueError, match="matching indices"):
        backtest_var(
            pd.Series([0.01, 0.02], index=index),
            pd.Series([0.0, 0.0], index=index + pd.Timedelta(days=1)),
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        backtest_var(pd.Series(dtype=float), pd.Series(dtype=float))


@pytest.mark.parametrize("rate", [-0.01, 1.01, np.nan])
def test_classify_backtest_exceptions_rejects_invalid_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="exception_rate"):
        classify_backtest_exceptions(rate)
