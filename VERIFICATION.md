# Market Risk Verification Summary

This summary documents the frozen analysis and local verification completed on 6 September 2026. The price snapshot contains 4,695 observations from 2 January 2008 through 31 August 2026. The portfolio uses SPY, TLT, GLD, XLE and EEM with constant target weights of 40%, 25%, 15%, 10% and 10%, respectively. Returns are daily simple returns, assuming frictionless daily rebalancing.

## Main finding

**An aggregate exception rate close to the expected rate did not establish adequate conditional coverage in this sample.**

| Diagnostic | Reproduced result | Interpretation |
|---|---:|---|
| Backtest period | 2009-01-02 to 2026-08-31 | Each one-day VaR forecast uses the preceding 252 daily returns |
| Exceptions / forecast days | 230 / 4,442 | Counted from the dated exception series |
| Exception rate | 5.17785% | Expected rate: 5% |
| Kupiec proportion-of-failures test | p = 0.588607 | Unconditional coverage is not rejected at the 5% significance level |
| Christoffersen independence test | p = 0.002332 | First-order exception independence is rejected |
| Combined conditional coverage test | p = 0.008394 | The joint null of correct coverage and independence is rejected |

The next-day exception rate was 23/230 = 10.0% following an exception, compared with 207/4,211 = approximately 4.92% following a non-exception. These sample diagnostics indicate serial dependence in exceptions. They do not establish overall model validity or constitute an official Basel classification.

## Other analytical results

- Historical 95% VaR / ES: -1.13346% / -1.88012%.
- Gaussian 95% VaR / ES: -1.28122% / -1.61604%.
- In this sample, Gaussian VaR has a larger loss magnitude than historical VaR, while Gaussian ES has a smaller loss magnitude than historical ES. Values use the signed-return convention: more negative means a larger loss.
- The first three principal components explain 89.6599% of the variance of standardized asset returns. This is not a decomposition of weighted portfolio variance.
- The realized lower 5th percentile during the COVID window is -5.95945%, or 5.25778 times the full-sample loss magnitude. Because the full sample includes the COVID window, this ratio is descriptive and is not a measure of forecast error.

Earlier README figures, including 231/4,370 exceptions, a 5.29% exception rate and -1.47% VaR, could not be reproduced from identical inputs because the original frozen data were unavailable. The current figures use a new dated snapshot and checked calculations. Differences are not attributed solely to a change in the end date.

## Changes made

1. Stored the raw Yahoo response, adjusted-close CSV, analysis configuration and file hashes. Subsequent analysis runs offline.
2. Removed the custom 5%/8% exception-rate bands previously labelled as Basel classifications.
3. Corrected PCA and crisis-window VaR labels to match the underlying calculations.
4. Included initial capital before the first return in crisis-window drawdown calculations, so a first-day loss contributes to drawdown.
5. Added Kupiec and Christoffersen tests and exception transition counts.
6. Executed all five code cells in the current analysis notebook and saved their outputs.
7. Locked the verified Python 3.12 package environment using a compatible dependency set.
8. Configured GitHub Actions to run tests against frozen data and separately regenerate and compare numerical results. At the time of this verification, the workflow commands had been checked locally; a remote Actions run had not been completed.

## Verification evidence

- Original project: all 101 existing tests passed in the new execution environment.
- Revised project: all 120 tests passed with network access blocked.
- Kupiec test: checked against a published reference example.
- Independence test: checked against manually tractable transition-count examples.
- Frozen snapshot: independently recalculated returns, prior-window quantiles, exception counts and test statistics using separate formulas.
- PCA: cross-checked against correlation-matrix eigenvalues.
- Data extraction: confirmed agreement between the raw response and the prepared adjusted-close CSV.
- Reproducibility: reran the analysis into a separate output directory and confirmed agreement of headline results.
- Visual review: inspected all four generated charts.

The regression baselines were saved from the corrected implementation. They detect changes but do not independently prove that the formulas are valid; separate reference and formula checks provide complementary evidence. This verification does not establish accuracy on future data or under all market conditions.

## Concise description of the finding

> Evaluated rolling 95% historical VaR over 4,442 trading days; found a 5.18% exception rate but rejected first-order exception independence (p=0.0023), highlighting limitations beyond aggregate coverage.

This description depends on the inputs, assumptions and tests documented here. The verification covers the current five-ETF analysis, not the separate legacy NVDA notebook or the momentum project.

## Supporting files

- [Full results and charts](results/REPORT.md)
- [Exact numerical results](results/summary.json)
- [Daily forecasts, realized returns and exceptions](results/daily_backtest.csv)
- [Data provenance and hashes](data/snapshot_2026-08-31/manifest.json)
- [Executed notebook](stress_testing.ipynb)
- [Reproduction instructions](README.md)
