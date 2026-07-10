"""Deterministic tests for historical stress calculations."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.risk_engine.stress import (
    calculate_cumulative_return,
    calculate_max_drawdown,
    run_stress_scenario,
)


RETURNS = np.array([0.10, -0.20, 0.05], dtype=float)


def test_calculate_cumulative_return_matches_independent_compounding() -> None:
    expected = (1.10 * 0.80 * 1.05) - 1.0

    assert calculate_cumulative_return(RETURNS) == pytest.approx(expected)


def test_calculate_max_drawdown_matches_independent_local_peak_calculation() -> None:
    cumulative_curve = np.array([1.10, 0.88, 0.924])
    rolling_max = np.maximum.accumulate(cumulative_curve)
    expected = float(((cumulative_curve - rolling_max) / rolling_max).min())

    assert calculate_max_drawdown(RETURNS) == pytest.approx(expected)


def test_run_stress_scenario_preserves_inclusive_window_and_notebook_metrics() -> None:
    portfolio_returns = pd.Series(
        [0.10, -0.20, 0.05, 0.01, -0.10],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    window = portfolio_returns.iloc[1:4]

    expected_cumulative = float(np.prod(1.0 + window.to_numpy()) - 1.0)
    cumulative_curve = np.cumprod(1.0 + window.to_numpy())
    expected_drawdown = float(
        ((cumulative_curve - np.maximum.accumulate(cumulative_curve))
         / np.maximum.accumulate(cumulative_curve)).min()
    )
    expected_var = float(np.percentile(window, 5.0))

    result = run_stress_scenario(
        portfolio_returns,
        start_date="2020-01-02",
        end_date=date(2020, 1, 4),
    )

    pd.testing.assert_series_equal(result["returns"], window)
    assert result["cum_ret"] == pytest.approx(expected_cumulative)
    assert result["max_dd"] == pytest.approx(expected_drawdown)
    assert result["var"] == pytest.approx(expected_var)


@pytest.mark.parametrize("function", [calculate_cumulative_return, calculate_max_drawdown])
def test_stress_calculations_reject_empty_nan_and_insufficient_inputs(function) -> None:
    for invalid in [np.array([]), np.array([0.01]), np.array([0.01, np.nan])]:
        with pytest.raises(ValueError):
            function(invalid)


def test_run_stress_scenario_rejects_invalid_return_series() -> None:
    for invalid in [
        pd.Series(dtype=float, index=pd.DatetimeIndex([])),
        pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1)),
        pd.Series([0.01, np.nan], index=pd.date_range("2020-01-01", periods=2)),
    ]:
        with pytest.raises(ValueError):
            run_stress_scenario(invalid, "2020-01-01", "2020-01-03")


@pytest.mark.parametrize(
    ("start_date", "end_date", "message"),
    [
        ("not-a-date", "2020-01-03", "start_date"),
        ("2020-01-03", "not-a-date", "end_date"),
        ("2020-01-04", "2020-01-03", "on or before"),
        ("2021-01-01", "2021-01-03", "no observations"),
    ],
)
def test_run_stress_scenario_rejects_invalid_or_missing_date_ranges(
    start_date: str, end_date: str, message: str
) -> None:
    returns = pd.Series(
        [0.01, -0.02],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    with pytest.raises(ValueError, match=message):
        run_stress_scenario(returns, start_date, end_date)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_run_stress_scenario_rejects_invalid_confidence(confidence: float) -> None:
    returns = pd.Series(
        [0.01, -0.02],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    with pytest.raises(ValueError, match="confidence"):
        run_stress_scenario(returns, "2020-01-01", "2020-01-02", confidence)
