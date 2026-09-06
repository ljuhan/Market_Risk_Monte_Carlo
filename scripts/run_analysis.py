"""Reproduce risk results from the hash-verified local snapshot, without network."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.risk_engine.snapshot import analyse_snapshot


def pct(value):
    return f"{value:.2%}"


def report(summary):
    bt = summary["backtest"]
    tests = ["kupiec_pof", "christoffersen_independence", "christoffersen_conditional_coverage"]
    lines = ["# Frozen-sample Market Risk Results", "",
             f"Price observations: **{summary['data']['actual_start']} to {summary['data']['actual_end']}**, {summary['data']['observation_count']:,} dates.",
             f"Retrieved: {summary['data']['extracted_at_utc']}. These replace the unverified historical README snapshot; they were not fitted to its numbers.", "",
             "## Rolling 95% historical VaR", "",
             f"**{bt['exceptions']:,} exceptions / {bt['observations']:,} forecast days = {pct(bt['exception_rate'])}**, versus {pct(bt['expected_exception_rate'])} expected.",
             f"Forecast dates: {bt['start']} to {bt['end']}. Each forecast uses only the preceding {bt['estimation_window']} returns.", "",
             "| Test | LR statistic | p-value | Decision at 5% |",
             "|---|---:|---:|---|"]
    for name in tests:
        t = bt[name]
        lines.append(f"| {name} | {t['lr_statistic']:.6f} | {t['p_value']:.6g} | {t['status']} |" if t['p_value'] is not None else f"| {name} | N/A | N/A | not identifiable |")
    lines += ["", "The POF null is a 5% unconditional exception probability; independence tests first-order dependence in adjacent hits. Combined coverage tests both restrictions. Failure to reject is not proof of accuracy; tests are asymptotic and do not establish regulatory compliance.",
              f"Transitions: `{bt['transition_counts']}`. A 252-day rolling exception plot is descriptive, not a Basel traffic light.", ""]
    lines += [f"- {w}" for w in bt['warnings']]
    lines += ["", "## Full-sample descriptive tail estimates", "",
              "Signed daily returns: more negative means a larger loss. Estimates use the same full sample; they are not forward forecasts.", "",
              "| Method | 95% VaR | 95% ES |", "|---|---:|---:|"]
    for name, m in summary["full_sample_methods"].items():
        lines.append(f"| {name} | {pct(m['var'])} | {pct(m['expected_shortfall'])} |")
    lines += ["", "## Realized stress windows", "",
              "| Scenario | Days | Cumulative return | Drawdown from initial/local peak | Window 95% VaR | / full-sample magnitude |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, s in summary["stress_scenarios"].items():
        lines.append(f"| {name} | {s['observations']} | {pct(s['cumulative_return'])} | {pct(s['max_drawdown'])} | {pct(s['realized_window_var'])} | {s['magnitude_ratio_to_full_sample_var']:.2f}x |")
    lines += ["", "These compare quantiles computed within historical windows against a full-sample quantile that includes the crises. A ratio is NOT evidence of an ex-ante forecast miss. The short COVID window makes its empirical tail quantile particularly imprecise. Drawdown includes capital before the first window return, but not peaks before the window.", "",
              "## PCA of standardized asset returns", "",
              f"The first two components explain **{pct(summary['pca']['first_two_cumulative'])}** and the first three **{pct(summary['pca']['first_three_cumulative'])}** of standardized asset-return variance.",
              "This is the five-asset correlation structure. Portfolio weights are not PCA inputs, so this is not a decomposition of portfolio variance. Component signs and economic interpretations are not identified uniquely.", "",
              "## What is and is not validated", "",
              "- Raw vendor response, adjusted-close CSV, immutable hashes, parameters, numerical results and dated exception series are stored together.",
              "- Regression tests check frozen headline results; separate formula tests check coverage calculations. Reproducibility is not proof of financial validity.",
              "- Fixed weights are rebalanced daily without costs. Asset selection/weights are retrospectively specified; vendor data were acquired after the sample end. The rolling calculation excludes the forecast return but is not a point-in-time production record.",
              "- No arbitrary 5%/8% regulatory classification remains in the current workflow. Historical audit notes are preserved as dated records, not current methodology.",
              "- No resume claims should imply a production deployment, official model approval, strategy alpha, or causal evidence from descriptive stress/PCA analysis.", "",
              "## Outputs", "",
              "![Backtest](backtest.png)", "", "![Tail comparison](tail_comparison.png)", "", "![PCA](pca.png)", "", "![Stress](stress.png)", "",
              "Sources: [Kupiec POF](https://www.mathworks.com/help/risk/varbacktest.pof.html), [Christoffersen independence](https://www.mathworks.com/help/risk/varbacktest.cci.html), [conditional coverage](https://www.mathworks.com/help/risk/varbacktest.cc.html).", ""]
    return "\n".join(lines)


def figures(summary, tables, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    daily = tables["daily_backtest"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, layout="constrained")
    axes[0].plot(daily.index, daily.realized_return * 100, color="#bac4ce", lw=.5, label="Realized daily return")
    axes[0].plot(daily.index, daily.forecast_var * 100, color="#184e77", lw=.8, label="Prior-window 95% historical VaR")
    hit = daily[daily.exception == 1]
    axes[0].scatter(hit.index, hit.realized_return * 100, s=5, color="#b64635", label="Exception")
    axes[0].set(ylabel="Daily return (%)", title="Rolling forecasts use only preceding 252 returns")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].plot(daily.index, daily.rolling_exception_rate * 100, color="#184e77", lw=1)
    axes[1].axhline(5, color="#b64635", ls="--", label="Expected rate: 5% (not a regulatory threshold)")
    axes[1].set(ylabel="252-day exception rate (%)", xlabel="Forecast date")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "backtest.png", dpi=150)
    plt.close(fig)
    methods = pd.DataFrame(summary["full_sample_methods"]).T
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    (-methods * 100).plot.bar(ax=ax, color=["#184e77", "#d48a36"], rot=0)
    ax.set(ylabel="Daily loss magnitude (%)", title="Full-sample tail estimates at 95% confidence")
    ax.legend(["VaR", "Expected shortfall"])
    fig.savefig(output / "tail_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    ratio = tables["pca_ratio"]
    axes[0].bar(ratio.index, ratio * 100, color="#184e77")
    axes[0].plot(ratio.index, ratio.cumsum() * 100, "o-", color="#d48a36", label="Cumulative")
    axes[0].set(ylabel="Standardized asset-return variance (%)", ylim=(0, 105), title="Correlation-structure PCA")
    axes[0].legend()
    coefficients = tables["pca_coefficients"]
    im = axes[1].imshow(coefficients, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_xticks(range(5), coefficients.columns)
    axes[1].set_yticks(range(5), coefficients.index)
    axes[1].set_title("Component coefficients (sign arbitrary)")
    fig.colorbar(im, ax=axes[1], shrink=.8)
    fig.savefig(output / "pca.png", dpi=150)
    plt.close(fig)
    stress = pd.DataFrame(summary["stress_scenarios"]).T
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    stress[["cumulative_return", "max_drawdown"]].astype(float).mul(100).plot.bar(ax=ax, color=["#184e77", "#d48a36"], rot=0)
    ax.set(ylabel="Return / drawdown (%)", title="Descriptive realized stress windows; not prediction errors")
    ax.legend(["Cumulative return", "Local drawdown incl. initial capital"])
    fig.savefig(output / "stress.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary, tables = analyse_snapshot(ROOT)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, data in tables.items():
        data.to_csv(output / f"{name}.csv", float_format="%.17g")
    pd.DataFrame(summary["stress_scenarios"]).T.to_csv(output / "stress_summary.csv")
    environment = {"python": platform.python_version(), "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pytest", "yfinance"]}}
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    figures(summary, tables, output)
    (output / "REPORT.md").write_text(report(summary))
    print(json.dumps(summary["backtest"], indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
