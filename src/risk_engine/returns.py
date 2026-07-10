"""Return and portfolio-weight calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _as_numeric_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert values to a finite one-dimensional float array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_returns(asset_returns: pd.DataFrame) -> None:
    """Validate a non-empty numeric return DataFrame."""
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame")
    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty")
    if asset_returns.shape[1] == 0:
        raise ValueError("asset_returns must contain at least one asset")
    if not all(is_numeric_dtype(dtype) for dtype in asset_returns.dtypes):
        raise ValueError("asset_returns must contain only numeric values")
    values = asset_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("asset_returns must contain only finite values")


def calculate_simple_returns(
    prices: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Calculate simple percentage returns with pandas ``pct_change`` semantics."""
    if not isinstance(prices, (pd.DataFrame, pd.Series)):
        raise TypeError("prices must be a pandas DataFrame or Series")
    if prices.empty:
        raise ValueError("prices cannot be empty")
    if not all(is_numeric_dtype(dtype) for dtype in prices.dtypes) if isinstance(prices, pd.DataFrame) else not is_numeric_dtype(prices.dtype):
        raise ValueError("prices must contain only numeric values")

    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prices must contain only finite values")
    if (values <= 0).any():
        raise ValueError("prices must contain only positive values")
    if len(prices) < 2:
        raise ValueError("prices must contain at least two observations")

    return prices.pct_change().dropna()


def validate_portfolio_weights(
    tickers: Sequence[str],
    weights: Sequence[float],
) -> None:
    """Validate ticker and portfolio-weight alignment and unit total."""
    if not isinstance(tickers, Sequence) or isinstance(tickers, (str, bytes)):
        raise TypeError("tickers must be a sequence of ticker names")
    if len(tickers) == 0:
        raise ValueError("tickers cannot be empty")
    if any(not isinstance(ticker, str) or not ticker for ticker in tickers):
        raise ValueError("tickers must contain non-empty strings")
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must not contain duplicates")

    weight_array = _as_numeric_array(weights, "weights")
    if len(tickers) != len(weight_array):
        raise ValueError("tickers and weights must have the same length")
    if not np.isclose(weight_array.sum(), 1.0):
        raise ValueError("portfolio weights must sum to 1")


def calculate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    """Calculate portfolio returns as the weighted sum of asset returns."""
    _validate_returns(asset_returns)
    validate_portfolio_weights(list(asset_returns.columns), weights)
    return asset_returns.mul(np.asarray(weights, dtype=float), axis=1).sum(axis=1)
