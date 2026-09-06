"""One-time Yahoo download; subsequent analysis is entirely offline.

Never overwrite a snapshot. Use a new directory/config for a new snapshot.
"""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf
from src.risk_engine.data import validate_price_data


def main():
    config_path = ROOT / "analysis_config.json"
    config = json.loads(config_path.read_text())
    target = ROOT / "data" / "snapshot_2026-08-31"
    if target.exists():
        raise SystemExit(f"Refusing to overwrite immutable snapshot: {target}")
    end = pd.Timestamp(config["end_date_inclusive"])
    exclusive_end = (end + timedelta(days=1)).strftime("%Y-%m-%d")
    acquired = datetime.now(timezone.utc).isoformat()
    yf.set_tz_cache_location(str(ROOT / ".cache" / "yfinance"))
    # A new download can have historical revisions; its raw bytes are retained.
    raw = yf.download(config["tickers"], start=config["start_date_inclusive"],
                      end=exclusive_end, auto_adjust=False, actions=True,
                      threads=False, progress=False, timeout=30)
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("No complete Yahoo response; no snapshot created")
    adjusted = raw["Adj Close"].loc[:, config["tickers"]].copy()
    if adjusted.index.tz is not None:
        adjusted.index = adjusted.index.tz_localize(None)
    adjusted.index = adjusted.index.normalize()
    adjusted = adjusted.sort_index()
    adjusted = adjusted.loc[config["start_date_inclusive"]:config["end_date_inclusive"]]
    clean = validate_price_data(adjusted, config["tickers"])
    if clean.index[-1] != end:
        raise RuntimeError(f"Expected final trading date {end.date()}, got {clean.index[-1].date()}; refusing incomplete snapshot")
    # 2026-08-31 is a US trading session. Confirm each series covers 2008.
    if clean.index[0] > pd.Timestamp("2008-01-03"):
        raise RuntimeError("Historical response does not cover the requested start")
    target.mkdir(parents=True)
    raw.to_csv(target / "raw_yahoo.csv", float_format="%.17g")
    clean.to_csv(target / "adjusted_close.csv", index_label="Date", float_format="%.17g", date_format="%Y-%m-%d")
    files = {name: sha256((target / name).read_bytes()).hexdigest()
             for name in ["raw_yahoo.csv", "adjusted_close.csv"]}
    manifest = {
        "source": "Yahoo Finance historical prices via yfinance; Adj Close selected from auto_adjust=False response",
        "source_urls": [f"https://finance.yahoo.com/quote/{t}/history/" for t in config["tickers"]],
        "extracted_at_utc": acquired,
        "requested_start_inclusive": config["start_date_inclusive"],
        "requested_end_inclusive": config["end_date_inclusive"],
        "yfinance_end_exclusive": exclusive_end,
        "actual_start": str(clean.index[0].date()),
        "actual_end": str(clean.index[-1].date()),
        "observation_count": len(clean),
        "tickers": config["tickers"],
        "missing_rows_removed": [str(d.date()) for d in adjusted.index.difference(clean.index)],
        "timezone": "Date-only US exchange sessions (America/New_York); timezone removed without date conversion",
        "cleaning": "Select Adj Close; retain configured ticker order; sort ascending; restrict inclusive dates; drop rows missing any of five assets; no forward fill",
        "limitations": "Vendor-adjusted historical snapshot acquired after the sample end; not point-in-time vendor data. Later vendor revisions are not applied to this snapshot.",
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "files_sha256": files,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "yfinance"]},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
