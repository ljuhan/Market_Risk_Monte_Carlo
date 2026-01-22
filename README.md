# Monte Carlo Value-at-Risk (VaR) Engine

## Overview
This project is a **Quantitative Risk Model** designed to estimate the potential downside risk of an asset using **Monte Carlo Simulations**. 

It implements **Geometric Brownian Motion (GBM)** to generate 10,000+ stochastic price paths and calculates key risk metrics:
* **Value at Risk (VaR 95%):** The threshold of max loss with 95% confidence.
* **Expected Shortfall (CVaR 95%):** The average loss in the worst 5% of scenarios (tail risk).

## Features
* **Vectorized Implementation:** Uses `NumPy` linear algebra instead of loops for high-performance simulation (capable of generating 100,000+ paths in seconds).
* **Automated Data Pipeline:** Fetches real-time historical data using the `yfinance` API.
* **Dynamic Parameterization:** Automatically calculates Drift ($\mu$) and Volatility ($\sigma$) based on historical log-returns.
* **Visualizations:**
    * Monte Carlo Price Paths (Spaghetti Plot).
    * Return Distribution Histograms with VaR/CVaR thresholds.

## Tech Stack
* **Python 3.10+**
* **NumPy** (Stochastic generation & vectorization)
* **Pandas** (Time-series data handling)
* **Matplotlib** (Data visualization)
* **YFinance** (Market data API)

## How to Run
1. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib yfinance