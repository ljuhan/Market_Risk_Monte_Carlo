"""Frozen market-data regression AND independent checks on its calculations."""
import json
import math
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
import yfinance
from src.risk_engine.snapshot import analyse_snapshot, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def analysis():
    # The offline path must never call the vendor.
    original = yfinance.download
    def no_download(*args, **kwargs):
        raise AssertionError("Frozen analysis attempted a network download")
    yfinance.download = no_download
    try:
        yield analyse_snapshot(ROOT)
    finally:
        yfinance.download = original


def compare(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-10, rel=1e-9)
    else:
        assert actual == expected


def test_frozen_headline_numbers_match_reviewed_baseline(analysis):
    expected = json.loads((ROOT / "tests/fixtures/expected_summary.json").read_text())
    compare(analysis[0], expected)


def test_committed_summary_matches_current_computation(analysis):
    committed = json.loads((ROOT / "results/summary.json").read_text())
    compare(analysis[0], committed)


def test_clean_csv_is_adjusted_close_from_raw_vendor_response():
    config, manifest, clean = load_snapshot(ROOT)
    path = ROOT / "data/snapshot_2026-08-31/raw_yahoo.csv"
    raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True, float_precision="round_trip")
    expected = raw["Adj Close"].loc[:, config["tickers"]].dropna()
    pd.testing.assert_frame_equal(clean, expected, check_names=False)


def test_returns_forecasts_and_hits_recomputed_independently(analysis):
    s, tables = analysis
    p = tables["prices"]
    weights = np.array([.4, .25, .15, .1, .1])
    r = pd.Series(((p.to_numpy()[1:] / p.to_numpy()[:-1] - 1) @ weights), index=p.index[1:])
    expected_forecast = r.rolling(252).quantile(.05).shift(1).dropna()
    daily = tables["daily_backtest"]
    np.testing.assert_allclose(daily.forecast_var, expected_forecast, rtol=1e-12, atol=1e-15)
    assert daily.index.equals(expected_forecast.index)
    np.testing.assert_array_equal(daily.exception, (r.loc[daily.index] < expected_forecast).astype(int))
    assert daily.exception.sum() == s["backtest"]["exceptions"]


def test_pca_matches_independent_correlation_eigenvalues(analysis):
    s, t = analysis
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(t["asset_returns"].to_numpy(), rowvar=False))[::-1]
    np.testing.assert_allclose(s["pca"]["explained_variance_ratio"], eigenvalues / eigenvalues.sum(), atol=1e-12)


def test_pof_and_transition_likelihoods_independent_of_coverage_implementation(analysis):
    s, tables = analysis
    hits = tables["daily_backtest"].exception.to_numpy()
    n, x, p = len(hits), int(sum(hits)), .05
    lr_uc = 2 * (x * math.log((x/n)/p) + (n-x) * math.log((1-x/n)/(1-p)))
    counts = [sum((hits[:-1] == a) & (hits[1:] == b)) for a, b in [(0,0), (0,1), (1,0), (1,1)]]
    a,b,c,d = counts
    p0, p1, pooled = b/(a+b), d/(c+d), (b+d)/(n-1)
    alt = a*math.log(1-p0)+b*math.log(p0)+c*math.log(1-p1)+d*math.log(p1)
    null = (a+c)*math.log(1-pooled)+(b+d)*math.log(pooled)
    lr_ind = 2*(alt-null)
    bt = s["backtest"]
    assert bt["kupiec_pof"]["p_value"] == pytest.approx(math.erfc(math.sqrt(lr_uc/2)))
    assert bt["christoffersen_independence"]["p_value"] == pytest.approx(math.erfc(math.sqrt(lr_ind/2)))
    assert bt["christoffersen_conditional_coverage"]["p_value"] == pytest.approx(math.exp(-(lr_uc+lr_ind)/2))


def test_all_published_daily_rows_match_recomputation(analysis):
    actual = analysis[1]["daily_backtest"]
    stored = pd.read_csv(ROOT / "results/daily_backtest.csv", index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(actual, stored, check_names=False, check_freq=False, atol=1e-12, rtol=1e-10)


def test_snapshot_tampering_is_rejected(tmp_path):
    shutil.copy(ROOT / "analysis_config.json", tmp_path)
    shutil.copytree(ROOT / "data", tmp_path / "data")
    prices = tmp_path / "data/snapshot_2026-08-31/adjusted_close.csv"
    with prices.open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(tmp_path)
