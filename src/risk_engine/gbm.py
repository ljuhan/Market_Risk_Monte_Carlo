"""Geometric Brownian Motion simulation and terminal-price risk metrics."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd
import yfinance as yf


def _validate_scalar(value: float, name: str, *, positive: bool = False) -> float:
    """Validate a finite numeric scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_confidence(confidence: float) -> float:
    """Return a valid confidence level."""
    confidence = _validate_scalar(confidence, "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


def _validate_price_array(
    prices: pd.Series | np.ndarray | list[float],
    name: str,
    minimum_observations: int = 1,
) -> np.ndarray:
    """Validate a non-empty one-dimensional positive price array."""
    if isinstance(prices, pd.Series) and not pd.api.types.is_numeric_dtype(prices.dtype):
        raise ValueError(f"{name} must contain only numeric values")
    try:
        values = np.asarray(prices)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if values.size < minimum_observations:
        raise ValueError(
            f"{name} must contain at least {minimum_observations} observations"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must contain only numeric values")
    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if (values <= 0.0).any():
        raise ValueError(f"{name} must contain only positive values")
    return values


def _extract_prices(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract the notebook's adjusted close price series."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError(f"No data found for {ticker}: data is empty")

    working = data.copy()
    if isinstance(working.columns, pd.MultiIndex):
        working.columns = working.columns.get_level_values(0)
    if "Close" in working.columns:
        prices = working["Close"]
    elif "Adj Close" in working.columns:
        prices = working["Adj Close"]
    else:
        prices = working.iloc[:, 0]
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(f"price data for {ticker} must identify one price series")
        prices = prices.iloc[:, 0]
    return prices


def get_simulation_parameters(
    ticker: str,
    start_date: str = "2020-01-01",
    data: pd.DataFrame | None = None,
) -> tuple[float, float, float]:
    """Download or read prices and return annualized ``S0``, ``mu``, and ``sigma``."""
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("ticker must be a non-empty string")
    try:
        parsed_start = pd.Timestamp(start_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("start_date must be a valid date") from exc
    if pd.isna(parsed_start):
        raise ValueError("start_date must be a valid date")

    if data is None:
        data = yf.download(
            ticker,
            start=start_date,
            progress=False,
            auto_adjust=True,
        )
    prices = _extract_prices(data, ticker)
    price_values = _validate_price_array(prices, "prices", minimum_observations=3)
    log_returns = np.log(price_values[1:] / price_values[:-1])

    sigma = float(log_returns.std(ddof=1) * np.sqrt(252))
    mu = float(log_returns.mean() * 252)
    s0 = float(price_values[-1])
    return s0, mu, sigma


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    M: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate GBM paths with notebook-compatible ``(N + 1, M)`` shape."""
    s0 = _validate_scalar(S0, "S0", positive=True)
    drift_rate = _validate_scalar(mu, "mu")
    volatility = _validate_scalar(sigma, "sigma")
    if volatility < 0.0:
        raise ValueError("sigma must be non-negative")
    horizon = _validate_scalar(T, "T", positive=True)
    if isinstance(N, bool) or not isinstance(N, Integral) or N <= 0:
        raise ValueError("N must be a positive integer")
    if isinstance(M, bool) or not isinstance(M, Integral) or M <= 0:
        raise ValueError("M must be a positive integer")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, Integral)
    ):
        raise ValueError("seed must be an integer or None")

    N = int(N)
    M = int(M)
    dt = horizon / N
    rng = np.random.RandomState(None if seed is None else int(seed))
    z = rng.normal(0.0, 1.0, size=(N, M))
    drift = (drift_rate - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt) * z
    daily_returns = np.exp(drift + diffusion)

    price_paths = np.zeros((N + 1, M))
    price_paths[0] = s0
    price_paths[1:] = s0 * np.cumprod(daily_returns, axis=0)
    return pd.DataFrame(price_paths)


def calculate_gbm_var(
    S0: float,
    terminal_prices: pd.Series | np.ndarray | list[float],
    confidence: float = 0.95,
) -> float:
    """Calculate the notebook's terminal-price VaR loss amount."""
    s0 = _validate_scalar(S0, "S0", positive=True)
    prices = _validate_price_array(terminal_prices, "terminal_prices")
    confidence = _validate_confidence(confidence)
    threshold = np.percentile(prices, (1.0 - confidence) * 100.0)
    return float(s0 - threshold)


def calculate_empirical_cvar(
    S0: float,
    terminal_prices: pd.Series | np.ndarray | list[float],
    confidence: float = 0.95,
) -> float:
    """Calculate the average terminal-price loss at or below GBM VaR."""
    s0 = _validate_scalar(S0, "S0", positive=True)
    prices = _validate_price_array(terminal_prices, "terminal_prices")
    confidence = _validate_confidence(confidence)
    threshold = np.percentile(prices, (1.0 - confidence) * 100.0)
    tail_prices = prices[prices <= threshold]
    return float(s0 - tail_prices.mean())
