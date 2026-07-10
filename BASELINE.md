# Risk Engine Baseline

Baseline capture date: **2026-07-10**

This records current notebook behavior before modularization. It is a software baseline, not an approval of the financial methodology.

## Repository Files and Notebook Roles

- `README.md`: project description, methodology summary, stored results, and run instructions.
- `stress_testing.ipynb`: primary five-asset portfolio workflow for data download, VaR/CVaR, stress testing, PCA, and rolling VaR backtesting.
- `VaR Simulation.ipynb`: separate single-asset NVDA GBM simulation and analytical VaR comparison.
- `var_comparison.png`: VaR/CVaR comparison chart from `stress_testing.ipynb`.
- `stress_test.png`: crisis-period performance chart from `stress_testing.ipynb`.
- `pca_decomposition.png`: PCA scree plot and factor-loading heatmap from `stress_testing.ipynb`.
- `var_backtest.png`: rolling VaR backtest chart from `stress_testing.ipynb`.
- `.gitignore`: Git ignore rules.
- `.git`: repository metadata.
- `REFACTOR_PLAN.md`: audit and staged implementation plan.

The notebooks are the executable source of the current calculations. PNG files are stored output artifacts, not outputs of a separate Python package.

## Portfolio Configuration

| Ticker | Weight |
|---|---:|
| SPY | 0.40 |
| TLT | 0.25 |
| GLD | 0.15 |
| XLE | 0.10 |
| EEM | 0.10 |

The weights sum to 1.0. No separate validation function currently enforces this.

## Data Dates and End-Date Behavior

### Portfolio notebook

- Start date: `2008-01-01`.
- End date: `datetime.today().strftime('%Y-%m-%d')`.
- Source: yfinance with `auto_adjust=True`.
- Price field: `raw['Close']`.
- Missing-price policy: `dropna()` across all five assets.
- Returns begin after the first valid price row.

The yfinance `end` parameter is an end boundary, so the last included market date can be earlier than the requested calendar date. Data is live and not frozen in the repository.

### NVDA notebook

- Start date: `2020-01-01` by default in `get_simulation_parameters`.
- End date: not explicitly supplied; yfinance uses current-date behavior.
- Price field: `Close`, then `Adj Close`, then the first returned column as fallback.
- `S0` is the last available price, although it is printed as `Start Price`.

## Return Conventions

Portfolio individual simple return:

`r_i,t = P_i,t / P_i,t-1 - 1`

Portfolio daily return:

`r_p,t = sum_i(w_i * r_i,t)`

This is weighted daily-return aggregation, implicitly representing constant target weights or daily rebalancing. It is not a buy-and-hold calculation using evolving position values.

NVDA log return:

`l_t = ln(P_t / P_t-1)`

Annualized NVDA parameters use 252 trading days:

`sigma = sample_std(l_t) * sqrt(252)`

`mu = mean(l_t) * 252`

The current pandas/numpy defaults use sample standard deviation (`ddof=1`).

## Confidence, Simulation, and Randomness

- Confidence level: `0.95`.
- Tail probability: `alpha = 1 - confidence = 0.05`.
- Portfolio Monte Carlo: `100000` normal draws.
- Portfolio Monte Carlo seed: global NumPy seed `42`.
- NVDA GBM paths: `M = 100000`.
- NVDA GBM steps: `N = 252`.
- NVDA GBM horizon: `T = 1.0` year.
- NVDA GBM seed: none; results vary between executions.

The NVDA notebook has a stale comment saying `M = 10,000`, but the executed value is `100000`.

## Current Formulas and Sign Conventions

Portfolio VaR and CVaR are signed return values, normally negative for losses. The tail is the 5th percentile.

Historical:

`hist_var = percentile(r_p, 5)`

`hist_cvar = mean(r_p[r_p <= hist_var])`

Parametric normal:

`mu = mean(r_p)`

`sigma = sample_std(r_p)`

`param_var = norm.ppf(0.05, mu, sigma)`

With `z = norm.ppf(0.05)`, parametric CVaR is:

`param_cvar = mu - sigma * norm.pdf(z) / 0.05`

Portfolio Monte Carlo draws `mc_returns ~ Normal(mu, sigma)` using 100,000 draws and seed 42, then applies the same percentile and `<=` CVaR rules.

For each stress period `r_s`:

`cum_ret = product(1 + r_s) - 1`

`cum_curve = cumprod(1 + r_s)`

`rolling_max = cum_curve.cummax()`

`drawdown = (cum_curve - rolling_max) / rolling_max`

`max_dd = min(drawdown)`

`crisis_var = percentile(r_s, 5)`

Stress-test values are signed; losses are negative.

PCA first applies `StandardScaler` to every asset return column, fits unconstrained `sklearn.decomposition.PCA()`, records `pca.explained_variance_ratio_`, and uses `pca.components_.T` as loadings. Component signs are not financially fixed.

Rolling backtest parameters are `window = 252`, `confidence = 0.95`, and `alpha = 0.05`. For index `i` from 252 onward:

`window_returns = portfolio_returns.iloc[i-252:i]`

`rolling_var[i] = percentile(window_returns, 5)`

An exception is `actual_return < rolling_var`. The exception rate is the Boolean mean. The current classification uses rates at or below 5% as acceptable, above 5% through 8% as yellow, and above 8% as red, while also printing annual bands of 0-4, 5-9, and 10+ exceptions.

NVDA GBM uses:

`dt = T / N`

`drift = (mu - 0.5 * sigma^2) * dt`

`diffusion = sigma * sqrt(dt) * Z`

`daily_returns = exp(drift + diffusion)`

`price_paths[0] = S0`

`price_paths[1:] = S0 * cumprod(daily_returns, axis=0)`

The terminal threshold is `future_price_95 = percentile(terminal_prices, 5)`. NVDA loss amounts are positive:

`VaR_amount = S0 - future_price_95`

`VaR_percent = VaR_amount / S0`

`CVaR_price = mean(terminal_prices[terminal_prices <= future_price_95])`

`CVaR_loss = S0 - CVaR_price`

The analytical comparison uses `z = norm.ppf(0.05)`, `parametric_price = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*z)`, and `parametric_VaR = S0 - parametric_price`.

## Stress-Period Definitions

| Name | Start | End |
|---|---|---|
| 2008 Financial Crisis | `2008-09-01` | `2009-03-31` |
| COVID Crash | `2020-02-19` | `2020-03-23` |
| 2022 Rate Hikes | `2022-01-01` | `2022-12-31` |

Only available trading dates within each calendar range are used.

## Notebook Execution-Order Dependencies

The portfolio notebook must run top to bottom. Cell 1 creates the data and portfolio globals; cell 2 creates VaR globals; cell 3 uses them for the comparison chart; cell 4 creates `scenario_results`; cell 5 plots those results; cell 6 creates PCA outputs; cell 7 plots them; cell 8 creates rolling-backtest outputs; and cell 9 plots them.

The NVDA notebook also requires top-to-bottom execution. Cell 0 creates `get_simulation_parameters`, `S0`, `mu`, and `sigma`; cell 1 creates `simulate_gbm`; cell 2 creates `T`, `N`, `M`, and `price_paths`; cell 3 creates `confidence_level`, `future_price_95`, and `VaR_amount`; cell 4 depends on those globals and on `terminal_prices`. Cell 4 references `ticker`, which is not created, so the report normally falls back to `Asset`. Cell 5 depends on all earlier GBM and VaR globals.

Existing outputs can be stale relative to live market data. The stored portfolio snapshot previously covered 2008-01-02 through 2026-05-18, while the code's end date is dynamic.

## Current Methodological Limitations

- Live, unversioned market data prevents exact reproduction from the repository alone.
- Adjusted-price behavior and missing-data handling are implicit.
- Portfolio aggregation assumes constant target weights or daily rebalancing.
- Historical VaR assumes the sample remains representative.
- Normal models omit fat tails, volatility clustering, serial dependence, and changing correlations.
- Portfolio Monte Carlo simulates portfolio returns directly, not asset-level paths or nonlinear exposures.
- The notebooks use different return conventions and horizons.
- Stress drawdown is local to each scenario window and excludes a pre-window peak.
- Standardized-return PCA explains correlation-structure variance, not directly weighted portfolio variance.
- PCA economic labels are heuristic and component signs are arbitrary.
- Backtest rate thresholds and annual exception bands are simplified and not equivalent classifications.
- There are no transaction costs, slippage, liquidity, leverage, taxes, position constraints, confidence intervals, or formal coverage tests.

These limitations are recorded for later methodology review and should not change during the initial preservation refactor.

## Baseline Regeneration Date

Baseline outputs would need to be regenerated on **2026-07-10**, the baseline capture date, if the baseline is intended to represent the notebooks' active calendar-date behavior. Since the portfolio notebook uses `datetime.today()` and yfinance's end boundary, the actual last included trading session may be the most recent available session before that date. Regeneration should record the actual downloaded date range, package environment, random seeds, and generated output values.

