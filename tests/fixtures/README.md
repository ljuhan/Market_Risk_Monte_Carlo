# Frozen Fixture Plan

This directory is reserved for a future deterministic market-data fixture used by numerical regression tests.

## Planned Fixture

The fixture should contain a small, frozen set of adjusted prices or precomputed asset returns for the five portfolio tickers used by `stress_testing.ipynb`:

- SPY
- TLT
- GLD
- XLE
- EEM

It should cover enough observations to test portfolio aggregation, VaR/CVaR, crisis-window filtering, rolling VaR, backtesting, and PCA without requiring a live Yahoo Finance request. A separate compact single-asset fixture may later be added for the NVDA GBM workflow.

## Future Dataset Requirements

- Store exact observation dates and values.
- Record whether the fixture contains adjusted prices, simple returns, or log returns.
- Record source, extraction date, date range, timezone, and cleaning performed.
- Keep values immutable once numerical baselines depend on them.
- Use a format loadable without network access.
- Include repeated and tail observations to exercise percentile and CVaR behavior.
- Test missing-data and invalid-input behavior with synthetic inputs rather than silently modifying the market-data fixture.

No market data has been downloaded or added in this stage. This README only documents the future fixture design.

