"""Rolling historical VaR backtesting calculations."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


def _validate_series(values: pd.Series, name: str) -> None:
    """Validate a non-empty finite numeric Series."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if values.empty:
        raise ValueError(f"{name} cannot be empty")
    if not pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError(f"{name} must contain only numeric values")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_confidence(confidence: float) -> float:
    """Return a valid confidence level."""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("confidence must be a finite number between 0 and 1")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")
    return confidence


def rolling_historical_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.Series:
    """Calculate historical VaR from prior observations only."""
    _validate_series(returns, "returns")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    confidence = _validate_confidence(confidence)
    if len(returns) <= window:
        raise ValueError("returns must contain more observations than window")

    alpha = 1.0 - confidence
    predicted_var = []
    predicted_dates = []
    for index in range(window, len(returns)):
        historical_window = returns.iloc[index - window:index]
        predicted_var.append(np.percentile(historical_window, alpha * 100.0))
        predicted_dates.append(returns.index[index])

    return pd.Series(predicted_var, index=predicted_dates)


def classify_backtest_exceptions(exception_rate: float) -> str:
    """Classify a rate using the notebook's simplified thresholds."""
    if isinstance(exception_rate, bool) or not isinstance(exception_rate, Real):
        raise ValueError("exception_rate must be a finite number between 0 and 1")
    exception_rate = float(exception_rate)
    if not np.isfinite(exception_rate) or not 0.0 <= exception_rate <= 1.0:
        raise ValueError("exception_rate must be a finite number between 0 and 1")

    if exception_rate <= 0.05:
        return "acceptable"
    if exception_rate <= 0.08:
        return "yellow"
    return "red"


def backtest_var(
    actual_returns: pd.Series,
    predicted_var: pd.Series,
) -> dict[str, object]:
    """Compare actual returns with aligned VaR using a strict less-than rule."""
    _validate_series(actual_returns, "actual_returns")
    _validate_series(predicted_var, "predicted_var")
    if not actual_returns.index.equals(predicted_var.index):
        raise ValueError("actual_returns and predicted_var must have matching indices")

    exceptions = actual_returns < predicted_var
    exception_count = int(exceptions.sum())
    observation_count = int(len(exceptions))
    exception_rate = exception_count / observation_count

    return {
        "predicted_var": predicted_var.copy(),
        "exceptions": exceptions,
        "exception_count": exception_count,
        "observation_count": observation_count,
        "exception_rate": exception_rate,
        "classification": classify_backtest_exceptions(exception_rate),
    }
