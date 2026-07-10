"""Deterministic tests for market-data loading and validation."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.risk_engine import data as data_module
from src.risk_engine.data import load_adjusted_prices, validate_price_data


DATES = pd.date_range("2020-01-01", periods=3, freq="D")
TICKERS = ["SPY", "TLT"]


def test_load_adjusted_prices_preserves_complete_case_behavior() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, np.nan, 102.0], "TLT": [50.0, 51.0, 52.0]},
        index=DATES,
    )

    actual = load_adjusted_prices(TICKERS, "2020-01-01", "2020-01-03", data=prices)

    expected = prices.iloc[[0, 2]]
    pd.testing.assert_frame_equal(actual, expected)


def test_load_adjusted_prices_extracts_close_from_adjusted_download() -> None:
    columns = pd.MultiIndex.from_product([["Close", "Volume"], TICKERS])
    raw_data = pd.DataFrame(
        [
            [100.0, 50.0, 1000.0, 2000.0],
            [101.0, 51.0, 1100.0, 2100.0],
            [102.0, 52.0, 1200.0, 2200.0],
        ],
        index=DATES,
        columns=columns,
    )

    actual = load_adjusted_prices(TICKERS, "2020-01-01", "2020-01-03", data=raw_data)

    expected = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0], "TLT": [50.0, 51.0, 52.0]},
        index=DATES,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_load_adjusted_prices_downloads_with_adjusted_close_settings() -> None:
    downloaded = pd.DataFrame({"SPY": [100.0, 101.0]}, index=DATES[:2])
    download = Mock(return_value=downloaded)
    original_download = data_module.yf.download
    data_module.yf.download = download
    try:
        actual = load_adjusted_prices(["SPY"], "2020-01-01", "2020-01-03")
    finally:
        data_module.yf.download = original_download

    pd.testing.assert_frame_equal(actual, downloaded)
    download.assert_called_once_with(
        ["SPY"], start="2020-01-01", end="2020-01-03", auto_adjust=True
    )


@pytest.mark.parametrize(
    ("tickers", "start_date", "end_date", "message"),
    [
        ([], "2020-01-01", "2020-01-03", "tickers cannot be empty"),
        (["SPY", "SPY"], "2020-01-01", "2020-01-03", "duplicates"),
        (["SPY"], "2020-01-04", "2020-01-03", "on or before"),
        (["SPY"], "not-a-date", "2020-01-03", "start_date"),
    ],
)
def test_load_adjusted_prices_rejects_invalid_inputs(
    tickers: list[str], start_date: str, end_date: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_adjusted_prices(tickers, start_date, end_date, data=pd.DataFrame())


def test_validate_price_data_rejects_missing_ticker_columns() -> None:
    prices = pd.DataFrame({"SPY": [100.0, 101.0]}, index=DATES[:2])

    with pytest.raises(ValueError, match="missing ticker columns"):
        validate_price_data(prices, ["SPY", "TLT"])


@pytest.mark.parametrize(
    "prices",
    [
        pd.DataFrame(),
        pd.DataFrame({"SPY": ["bad", "101"]}, index=DATES[:2]),
        pd.DataFrame({"SPY": [100.0, 101.0]}, index=["2020-01-01", "2020-01-02"]),
        pd.DataFrame(
            {"SPY": [100.0, 101.0], "TLT": [50.0, 51.0]},
            index=pd.DatetimeIndex([DATES[0], DATES[0]]),
        ),
        pd.DataFrame(
            {"SPY": [np.nan, np.nan], "TLT": [50.0, 51.0]},
            index=DATES[:2],
        ),
    ],
)
def test_validate_price_data_rejects_invalid_or_empty_data(
    prices: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        validate_price_data(prices, TICKERS)


def test_load_adjusted_prices_rejects_empty_download() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_adjusted_prices(TICKERS, "2020-01-01", "2020-01-03", data=pd.DataFrame())
