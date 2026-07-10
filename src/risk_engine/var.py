"""Value-at-Risk calculations for one-dimensional return series."""

from __future__ import annotations

from numbers import Real
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm


def _validate_confidence(confidence: float) -> float:
    """Return a valid confidence level."""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("confidence must be a finite number between 0 and 1")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")
    return confidence


def _validate_returns(
    returns: Sequence[float] | np.ndarray | pd.Series,
    minimum_observations: int = 2,
) -> np.ndarray:
    """Convert returns to a finite one-dimensional array with enough data."""
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


def historical_var(
    returns: Sequence[float] | np.ndarray | pd.Series,
    confidence: float = 0.95,
) -> float:
    """Return the historical lower-tail return percentile."""
    values = _validate_returns(returns)
    alpha = 1.0 - _validate_confidence(confidence)
    return float(np.percentile(values, alpha * 100.0))


def historical_cvar(
    returns: Sequence[float] | np.ndarray | pd.Series,
    confidence: float = 0.95,
) -> float:
    """Return the mean return at or below the historical VaR threshold."""
    values = _validate_returns(returns)
    alpha = 1.0 - _validate_confidence(confidence)
    var = np.percentile(values, alpha * 100.0)
    return float(values[values <= var].mean())


def parametric_var(
    returns: Sequence[float] | np.ndarray | pd.Series,
    confidence: float = 0.95,
) -> float:
    """Return normal-distribution VaR using sample mean and standard deviation."""
    values = _validate_returns(returns)
    alpha = 1.0 - _validate_confidence(confidence)
    return float(norm.ppf(alpha, values.mean(), values.std(ddof=1)))


def parametric_cvar(
    returns: Sequence[float] | np.ndarray | pd.Series,
    confidence: float = 0.95,
) -> float:
    """Return normal-distribution expected shortfall using the notebook formula."""
    values = _validate_returns(returns)
    alpha = 1.0 - _validate_confidence(confidence)
    mean = values.mean()
    sigma = values.std(ddof=1)
    z_score = norm.ppf(alpha)
    return float(mean - sigma * norm.pdf(z_score) / alpha)


def monte_carlo_normal_var(
    returns: Sequence[float] | np.ndarray | pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 100_000,
    seed: int | None = 42,
) -> float:
    """Return VaR from seeded normal draws fitted to historical returns."""
    values = _validate_returns(returns)
    alpha = 1.0 - _validate_confidence(confidence)
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, int):
        raise ValueError("n_simulations must be a positive integer")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError("seed must be an integer or None")

    rng = np.random.RandomState(seed)
    simulated_returns = rng.normal(values.mean(), values.std(ddof=1), n_simulations)
    return float(np.percentile(simulated_returns, alpha * 100.0))
