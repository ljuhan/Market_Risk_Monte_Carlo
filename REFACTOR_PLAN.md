# Risk Engine Audit and Refactor Plan

## Scope and Current State

This repository is notebook-led and contains two distinct workflows:

- `stress_testing.ipynb`: the primary five-asset portfolio workflow.
- `VaR Simulation.ipynb`: a separate single-asset NVDA geometric Brownian motion workflow.

The initial refactor should preserve the existing financial formulas, parameters, return conventions, scenario windows, and numerical behavior. Financial-methodology improvements should be reviewed separately after the software refactor has been validated.

The portfolio notebook's stored results were generated from live Yahoo Finance data covering approximately 2008-01-02 through 2026-05-18. The code uses `datetime.today()` as its end date, so those results are a historical snapshot rather than a stable, reproducible baseline. The worktree also contained an existing modification to `stress_testing.ipynb`; that change is outside this plan and must be preserved.

## 1. Existing Workflow and Calculations

### Portfolio workflow

1. Import NumPy, pandas, yfinance, SciPy, matplotlib, seaborn, and scikit-learn.
2. Define five assets and fixed weights:
   - SPY: 40%
   - TLT: 25%
   - GLD: 15%
   - XLE: 10%
   - EEM: 10%
3. Download adjusted close prices from `2008-01-01` through the current date.
4. Drop rows with missing prices.
5. Calculate simple daily returns.
6. Calculate portfolio returns as the weighted sum of asset returns.
7. Calculate one-day 95% VaR and CVaR using:
   - historical simulation;
   - a parametric normal distribution;
   - 100,000 normal Monte Carlo draws.
8. Evaluate three historical crisis windows:
   - 2008-09-01 through 2009-03-31;
   - 2020-02-19 through 2020-03-23;
   - 2022-01-01 through 2022-12-31.
9. Calculate crisis cumulative return, local maximum drawdown, and 5th-percentile crisis VaR.
10. Standardize each asset return series and run PCA.
11. Calculate a rolling 252-day historical VaR backtest and count exceptions.
12. Save VaR, stress-test, PCA, and backtest charts as PNG files.

### NVDA GBM workflow

1. Download adjusted NVDA prices from `2020-01-01`.
2. Calculate log returns.
3. Annualize the mean return and volatility using 252 trading days.
4. Simulate 100,000 one-year GBM paths with 252 time steps.
5. Calculate simulated 95% VaR and CVaR.
6. Calculate analytical GBM VaR and compare it with the simulation.

## 2. Current Financial Assumptions

- Portfolio weights are fixed and sum to 100%.
- Portfolio VaR is a one-day measure.
- There are 252 trading days per year.
- Prices are adjusted prices, so yfinance's dividend and split adjustments are used.
- Complete-case data is used: a date is removed if any asset price is missing.
- The weighted daily-return calculation implicitly represents constant target weights or daily rebalancing.
- Historical VaR assumes the observed return history is representative.
- Parametric and Monte Carlo portfolio VaR assume independent, normally distributed returns.
- Monte Carlo uses the historical portfolio mean and standard deviation.
- GBM assumes constant annualized drift and volatility.
- No transaction costs, slippage, liquidity constraints, leverage, taxes, or position limits are modeled.
- Stress scenarios are predefined historical windows.
- PCA is run on standardized returns, so its explained variance describes the standardized-return correlation structure rather than directly measuring raw portfolio variance.

The exact live-data output is not reproducible without freezing the downloaded data and the data end date. The NVDA simulation also lacks a random seed and therefore changes between executions.

## 3. Functions to Extract

The portfolio notebook currently places most calculations directly in cells. The following reusable functions should be extracted:

- `load_adjusted_prices(tickers, start_date, end_date)`
- `calculate_simple_returns(prices)`
- `validate_portfolio_weights(tickers, weights)`
- `calculate_portfolio_returns(asset_returns, weights)`
- `historical_var(returns, confidence)`
- `historical_cvar(returns, confidence)`
- `parametric_var(returns, confidence)`
- `parametric_cvar(returns, confidence)`
- `monte_carlo_normal_var(returns, confidence, n_simulations, seed)`
- `calculate_cumulative_return(returns)`
- `calculate_max_drawdown(returns)`
- `run_stress_scenario(returns, start_date, end_date, confidence)`
- `run_pca(returns, standardize=True)`
- `rolling_historical_var(returns, window, confidence)`
- `backtest_var(actual_returns, predicted_var)`
- `classify_backtest_exceptions(exception_count, observation_count)`

The NVDA workflow should separately extract:

- `get_simulation_parameters`
- `simulate_gbm`
- `calculate_gbm_var`
- `calculate_empirical_cvar`

Plotting and formatted reporting should remain separate from numerical functions.

## 4. Inputs, Outputs, and Hidden Notebook State

### Inputs

- tickers and portfolio weights;
- start and end dates;
- confidence level;
- rolling window length;
- Monte Carlo simulation count and seed;
- GBM parameters, horizon, and number of time steps;
- stress-scenario date ranges.

### Outputs

- price and return DataFrames;
- portfolio return Series;
- VaR and CVaR values;
- stress-scenario metrics;
- PCA explained variance and loadings;
- backtest exception Series and rates;
- printed reports and four PNG charts.

### Hidden state dependencies

- The portfolio notebook relies on globals created in earlier cells, including `returns`, `portfolio_returns`, `tickers`, `weights`, `hist_var`, and `scenario_results`.
- `confidence` and `alpha` are reused across cells.
- The NVDA notebook relies on global `S0`, `mu`, `sigma`, `T`, `price_paths`, `future_price_95`, and `confidence_level`.
- Cell 4 of the NVDA notebook references `ticker`, but that variable is not created; the fallback prints `Asset`.
- Existing notebook outputs can be stale relative to the current live-data download.
- The portfolio Monte Carlo calculation is seeded with `42`, but its data is not frozen.
- The NVDA Monte Carlo calculation has no seed.
- Chart files are side effects and are not represented by a structured result object.

A clean-kernel, top-to-bottom execution check should be added before and after extraction.

## 5. Duplication and Possible Issues

### Software-engineering improvements

- VaR and CVaR calculations are repeated inline rather than centralized.
- Crisis return, drawdown, and percentile calculations are embedded in the scenario loop.
- Plotting cells repeat data preparation and formatting logic.
- Configuration values are scattered across cells.
- There is no explicit validation for weights, dates, missing tickers, or empty data.
- Dependencies are not pinned.
- Live market data prevents stable regression testing.
- The two notebooks use inconsistent return conventions: log returns for NVDA GBM and simple returns for the portfolio.

### Financial-methodology questions for a later review

1. PCA is run after standardization, so describing its explained variance as portfolio variance is potentially misleading.
2. PCA factor names are heuristic and can change with the sample; component signs are arbitrary.
3. The backtest uses aggregate 5% and 8% exception-rate thresholds over the full sample while also displaying annual Basel exception-count bands. These are not equivalent classifications.
4. The Monte Carlo portfolio model draws independent normal portfolio returns. It does not simulate asset-level paths, correlations, rebalancing, or nonlinear instruments.
5. Normality assumptions omit fat tails, volatility clustering, and changing correlations.
6. Stress-test drawdown is local to the selected window and does not include a portfolio peak before the window.
7. `dropna()` creates an implicit missing-data policy that should be made explicit.
8. In the NVDA notebook, `S0` is the latest available price, although the output labels it as a start price.
9. README results should identify the data snapshot date and parameter set.

These issues should not be changed during the initial extraction unless a separate methodology decision is approved.

## 6. Proposed Repository Structure

```text
src/
  risk_engine/
    __init__.py
    config.py
    data.py
    returns.py
    var.py
    stress.py
    factors.py
    backtest.py
    gbm.py
    plotting.py
    reporting.py
    types.py
```

Responsibilities:

- `config.py`: tickers, weights, dates, confidence levels, and simulation settings.
- `data.py`: market-data acquisition, date handling, and validation.
- `returns.py`: simple/log returns and portfolio aggregation.
- `var.py`: historical, parametric, and Monte Carlo VaR/CVaR.
- `stress.py`: scenario filtering and crisis metrics.
- `factors.py`: standardized PCA, explained variance, and loadings.
- `backtest.py`: rolling VaR and exception analysis.
- `gbm.py`: single-asset GBM functionality.
- `plotting.py`: chart generation only.
- `reporting.py`: formatted summaries and result presentation.
- `types.py`: result dataclasses or typed mappings.

The notebooks should eventually become thin orchestration and visualization layers.

## 7. Tests for Numerical Preservation

Use deterministic synthetic returns and fixed seeds for unit tests. Do not make tests dependent on live Yahoo Finance data.

Required coverage:

- weight validation and weighted portfolio-return aggregation;
- historical VaR percentile behavior;
- historical CVaR's current `<=` inclusion rule;
- parametric VaR against `scipy.stats.norm.ppf`;
- parametric CVaR against the current formula;
- seeded Monte Carlo reproducibility;
- Monte Carlo output size, percentile, and CVaR;
- GBM path shape and initial row equal to `S0`;
- seeded GBM reproducibility;
- crisis cumulative return, local maximum drawdown, and crisis VaR;
- rolling VaR excluding the current observation from the lookback window;
- strict `<` exception behavior;
- PCA explained variance and loadings on a fixed matrix;
- missing-data, empty-data, invalid-weight, and invalid-parameter errors;
- clean-kernel, top-to-bottom notebook execution.

The stored portfolio numbers can serve as an initial comparison snapshot, but a frozen price or return fixture is required for durable regression tests.

## 8. Staged Implementation Plan

### Stage 1: Baseline capture

Record current formulas, cell dependencies, package versions, data date, seeds, and representative outputs. Add no production code changes in this stage.

### Stage 2: Pure calculation tests

Add tests for the existing calculations using deterministic fixtures and edge cases. Capture current behavior before extraction.

### Stage 3: Extract data and return logic

Add `data.py` and `returns.py`. Preserve current data alignment, adjusted-price usage, simple-return calculation, and portfolio aggregation. Update only the relevant notebook calls and verify equality.

### Stage 4: Extract VaR calculations

Add `var.py` and replace the three portfolio VaR implementations. Preserve percentile, CVaR, normal-distribution, simulation-count, and seed behavior. Verify exact numerical results.

### Stage 5: Extract stress testing

Add `stress.py`. Preserve the existing scenario dates, cumulative-return calculation, local drawdown definition, and crisis percentile VaR.

### Stage 6: Extract PCA and backtesting

Add `factors.py` and `backtest.py`. Preserve standardization, rolling window size, current-window exclusion, percentile rule, and strict exception comparison.

### Stage 7: Extract GBM functionality

Add `gbm.py` and move the single-asset simulation and analytical comparison while preserving annualization and path-generation formulas.

### Stage 8: Separate plotting and reporting

Add `plotting.py` and `reporting.py`. Keep chart contents and numerical labels consistent with the current notebooks.

### Stage 9: Thin the notebooks

Centralize configuration, pass values explicitly, remove hidden global dependencies, and make clean-kernel execution repeatable.

### Stage 10: Methodology review

Only after preservation is verified, separately review PCA terminology, Basel classification, data versioning, missing-data policy, non-normal distributions, volatility dynamics, and alternative stress methodologies.

## Refactor Guardrails

- Do not change either notebook during the initial plan-only phase.
- During implementation, make one small concern-level change per reviewable commit or change set.
- Keep financial-methodology decisions separate from module extraction.
- Use explicit parameter passing instead of notebook globals.
- Freeze data or use test fixtures for numerical regression tests.
- Record any intentional numerical difference and its reason.
