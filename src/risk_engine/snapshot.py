"""Load immutable data, verify provenance, and compute the headline analysis."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .backtest import rolling_historical_var, backtest_var
from .coverage import coverage_tests
from .data import validate_price_data
from .factors import run_pca
from .returns import calculate_simple_returns, calculate_portfolio_returns
from .stress import run_stress_scenario
from .var import historical_var, historical_cvar, parametric_var, parametric_cvar, monte_carlo_normal_var


def load_snapshot(root):
    root = Path(root)
    config_path = root / "analysis_config.json"
    config = json.loads(config_path.read_text())
    directory = root / "data" / "snapshot_2026-08-31"
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(config_path.read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Config hash mismatch; create a new version for changed assumptions")
    for filename, expected in manifest["files_sha256"].items():
        if sha256((directory / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot hash mismatch: {filename}")
    prices = pd.read_csv(directory / "adjusted_close.csv", index_col="Date", parse_dates=True, float_precision="round_trip")
    clean = validate_price_data(prices, config["tickers"])
    if len(clean) != len(prices) or list(prices.columns) != config["tickers"]:
        raise ValueError("Frozen data requires cleaning or has a changed column order")
    if len(clean) != manifest["observation_count"]:
        raise ValueError("Observation count mismatch")
    if str(clean.index[0].date()) != manifest["actual_start"] or str(clean.index[-1].date()) != manifest["actual_end"]:
        raise ValueError("Snapshot date range mismatch")
    return config, manifest, clean


def analyse_snapshot(root):
    config, manifest, prices = load_snapshot(root)
    returns = calculate_simple_returns(prices)
    portfolio = calculate_portfolio_returns(returns, config["weights"])
    confidence = config["confidence"]
    forecast = rolling_historical_var(portfolio, config["estimation_window"], confidence)
    realized = portfolio.loc[forecast.index]
    bt = backtest_var(realized, forecast)
    tests = coverage_tests(bt["exceptions"], 1 - confidence, config["test_significance"])
    daily = pd.DataFrame({"realized_return": realized, "forecast_var": forecast,
                          "exception": bt["exceptions"].astype(int)})
    daily["rolling_exception_rate"] = daily["exception"].rolling(config["rolling_diagnostic_window"]).mean()
    pca = run_pca(returns)
    ratio = pca["explained_variance_ratio"]
    rng = np.random.RandomState(config["random_seed"])
    simulated = rng.normal(portfolio.mean(), portfolio.std(), config["monte_carlo_draws"])
    mcvar = monte_carlo_normal_var(portfolio, confidence, config["monte_carlo_draws"], config["random_seed"])
    methods = {
        "Historical": {"var": historical_var(portfolio, confidence), "expected_shortfall": historical_cvar(portfolio, confidence)},
        "Gaussian": {"var": parametric_var(portfolio, confidence), "expected_shortfall": parametric_cvar(portfolio, confidence)},
        "Monte Carlo (Gaussian draws)": {"var": mcvar, "expected_shortfall": float(simulated[simulated <= mcvar].mean())},
    }
    scenarios = {}
    for name, (start, end) in config["scenarios"].items():
        s = run_stress_scenario(portfolio, start, end, confidence)
        overlap = daily.loc[start:end]
        scenarios[name] = {
            "start": str(s["returns"].index[0].date()),
            "end": str(s["returns"].index[-1].date()),
            "observations": len(s["returns"]),
            "cumulative_return": s["cum_ret"], "max_drawdown": s["max_dd"],
            "realized_window_var": s["var"],
            "magnitude_ratio_to_full_sample_var": abs(s["var"] / methods["Historical"]["var"]),
            "backtest_observations": len(overlap),
            "backtest_exceptions": int(overlap["exception"].sum()) if len(overlap) else None,
            "backtest_exception_rate": float(overlap["exception"].mean()) if len(overlap) else None,
        }
    summary = {
        "schema_version": 1,
        "data": {k: manifest[k] for k in ["source", "extracted_at_utc", "actual_start", "actual_end", "observation_count", "files_sha256", "config_sha256"]},
        "return_observations": len(portfolio),
        "portfolio_convention": "Constant daily target weights; adjusted-close simple returns; frictionless daily rebalancing, not buy-and-hold or executable P&L",
        "confidence": confidence,
        "backtest": {"start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
                     "estimation_window": config["estimation_window"], **tests},
        "full_sample_methods": methods,
        "pca": {"input": "StandardScaler-transformed five-asset returns, without portfolio weights",
                "explained_variance_ratio": ratio.tolist(),
                "first_two_cumulative": float(ratio.iloc[:2].sum()),
                "first_three_cumulative": float(ratio.iloc[:3].sum())},
        "stress_scenarios": scenarios,
    }
    return summary, {"prices": prices, "asset_returns": returns, "portfolio_returns": portfolio,
                     "daily_backtest": daily, "pca_coefficients": pca["loadings"],
                     "pca_ratio": ratio, "correlation": returns.corr()}
