"""Historical stress-period calculations."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _validate_returns(
    returns: Sequence[float] | np.ndarray | pd.Series,
    minimum_observations: int = 2,
) -> np.ndarray:
    """Convert returns to a finite one-dimensional array."""
    if isinstance(returns, pd.Series):
        if not is_numeric_dtype(returns.dtype):
            raise ValueError("returns must contain only numeric values")
    elif isinstance(returns, np.ndarray) and not np.issubdtype(returns.dtype, np.number):
        raise ValueError("returns must contain only numeric values")

    try:
        values = np.asarray(returns)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must be a one-dimensional numeric sequence") from exc
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if values.size == 0:
        raise ValueError("returns cannot be empty")
    if values.size < minimum_observations:
        raise ValueError(
            f"returns must contain at least {minimum_observations} observations"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("returns must contain only numeric values")

    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    return values


def _validate_confidence(confidence: float) -> float:
    """Return a valid confidence level."""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("confidence must be a finite number between 0 and 1")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")
    return confidence


def _parse_date(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    """Parse a scenario boundary as a pandas timestamp."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date")
    return timestamp


def calculate_cumulative_return(
    returns: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calculate cumulative return by compounding simple returns."""
    values = _validate_returns(returns)
    return float(np.prod(1.0 + values) - 1.0)


def calculate_max_drawdown(
    returns: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calculate the minimum drawdown from the local cumulative-return peak."""
    values = _validate_returns(returns)
    cumulative_curve = np.cumprod(1.0 + values)
    rolling_max = np.maximum.accumulate(cumulative_curve)
    drawdown = (cumulative_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def run_stress_scenario(
    portfolio_returns: pd.Series,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Calculate notebook-equivalent metrics for an inclusive date window."""
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series")
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        raise ValueError("portfolio_returns must have a DatetimeIndex")
    if portfolio_returns.index.has_duplicates:
        raise ValueError("portfolio_returns must not have duplicate dates")
    if not portfolio_returns.index.is_monotonic_increasing:
        raise ValueError("portfolio_returns dates must be sorted ascending")

    _validate_returns(portfolio_returns)
    confidence = _validate_confidence(confidence)
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    mask = (portfolio_returns.index >= start) & (portfolio_returns.index <= end)
    scenario_returns = portfolio_returns.loc[mask]
    if scenario_returns.empty:
        raise ValueError("the requested date range contains no observations")
    _validate_returns(scenario_returns)

    return {
        "returns": scenario_returns,
        "cum_ret": calculate_cumulative_return(scenario_returns),
        "max_dd": calculate_max_drawdown(scenario_returns),
        "var": float(np.percentile(scenario_returns, (1.0 - confidence) * 100.0)),
    }
