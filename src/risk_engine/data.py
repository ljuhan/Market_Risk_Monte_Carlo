"""Market-data loading and validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.api.types import is_numeric_dtype

DateLike = str | date | pd.Timestamp


def _validate_tickers(tickers: Iterable[str]) -> list[str]:
    """Return validated, ordered ticker names."""
    if isinstance(tickers, (str, bytes)):
        raise TypeError("tickers must be an iterable of ticker names")
    try:
        ticker_list = list(tickers)
    except TypeError as exc:
        raise TypeError("tickers must be an iterable of ticker names") from exc
    if not ticker_list:
        raise ValueError("tickers cannot be empty")
    if any(not isinstance(ticker, str) or not ticker for ticker in ticker_list):
        raise ValueError("tickers must contain non-empty strings")
    if len(set(ticker_list)) != len(ticker_list):
        raise ValueError("tickers must not contain duplicates")
    return ticker_list


def _parse_date(value: DateLike, name: str) -> pd.Timestamp:
    """Parse a date argument as a pandas timestamp."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date")
    return timestamp


def _extract_close_prices(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close columns from downloaded or prepared data."""
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if raw_data.empty:
        raise ValueError("downloaded data cannot be empty")

    prices: pd.DataFrame | pd.Series = raw_data
    if isinstance(raw_data.columns, pd.MultiIndex):
        levels = [set(raw_data.columns.get_level_values(i)) for i in range(raw_data.columns.nlevels)]
        if "Close" in levels[0]:
            prices = raw_data["Close"]
        elif "Adj Close" in levels[0]:
            prices = raw_data["Adj Close"]
        elif "Close" in levels[1]:
            prices = raw_data.xs("Close", axis=1, level=1)
        else:
            raise ValueError("data must contain a Close or Adj Close price field")
    elif "Close" in raw_data.columns:
        prices = raw_data[["Close"]].rename(columns={"Close": tickers[0]})
    elif "Adj Close" in raw_data.columns:
        prices = raw_data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices


def validate_price_data(prices: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Validate and select complete-case adjusted prices for the tickers."""
    ticker_list = _validate_tickers(tickers)
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty:
        raise ValueError("price data cannot be empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("price data must have a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("price data must not contain duplicate dates")
    if any(ticker not in prices.columns for ticker in ticker_list):
        missing = [ticker for ticker in ticker_list if ticker not in prices.columns]
        raise ValueError(f"price data is missing ticker columns: {missing}")

    selected = prices.loc[:, ticker_list]
    if not all(is_numeric_dtype(dtype) for dtype in selected.dtypes):
        raise ValueError("price data must contain only numeric values")
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError("price data must contain only finite values or NaN")
    if (selected.dropna().to_numpy(dtype=float) <= 0).any():
        raise ValueError("price data must contain only positive values")

    complete_case = selected.dropna()
    if complete_case.empty:
        raise ValueError("price data has no complete-case rows")
    return complete_case


def load_adjusted_prices(
    tickers: Iterable[str],
    start_date: DateLike,
    end_date: DateLike,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load adjusted close prices or validate supplied downloaded data."""
    ticker_list = _validate_tickers(tickers)
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    if data is None:
        raw_data = yf.download(
            ticker_list,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
    else:
        raw_data = data

    prices = _extract_close_prices(raw_data, ticker_list)
    return validate_price_data(prices, ticker_list)
