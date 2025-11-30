# scripts/run_baselines.py
# Compute and plot the main classical baselines:
# - Equal-Weight Buy & Hold
# - Markowitz Minimum Variance Portfolio
#
# Outputs:
# - Train and test metrics for both strategies (CSV)
# - Equity curve plots (PNG)

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORT_FIG_DIR, REPORT_TABLE_DIR
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test, compute_returns
from src.baselines.equal_weight import equity_curve_equal_weight
from src.baselines.markowitz import (
    compute_min_variance_weights,
    equity_curve_markowitz,
)
from src.evaluation.metrics import simple_metrics


def main() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load prices and split into train / test ---
    prices = load_prices()
    prices_train, prices_test = split_train_test(prices)

    # ----------------------------------------------------------------------
    # Equal-Weight Buy & Hold baseline
    # ----------------------------------------------------------------------
    print("\n--- Equal-Weight baseline ---")

    equity_train_eq = equity_curve_equal_weight(prices_train)
    equity_test_eq = equity_curve_equal_weight(prices_test)

    metrics_train_eq = simple_metrics(equity_train_eq)
    metrics_test_eq = simple_metrics(equity_test_eq)

    pd.DataFrame([metrics_train_eq]).to_csv(
        REPORT_TABLE_DIR / "metrics_train_baseline.csv", index=False
    )
    pd.DataFrame([metrics_test_eq]).to_csv(
        REPORT_TABLE_DIR / "metrics_test_baseline.csv", index=False
    )

    plt.figure(figsize=(10, 5))
    equity_train_eq.plot(label="Train")
    equity_test_eq.plot(label="Test")
    plt.title("Equal-Weight Buy & Hold Baseline")
    plt.ylabel("Portfolio value (start = 1.0)")
    plt.legend()
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "baseline_equal_weight.png")

    print("Train metrics (Equal-Weight):", metrics_train_eq)
    print("Test metrics  (Equal-Weight):", metrics_test_eq)

    # ----------------------------------------------------------------------
    # Markowitz Minimum Variance Portfolio
    # ----------------------------------------------------------------------
    print("\n--- Markowitz Minimum Variance Portfolio ---")

    returns_train = compute_returns(prices_train)
    weights_mvp = compute_min_variance_weights(returns_train)
    print("Optimal Markowitz weights:", weights_mvp)

    equity_train_mvp = equity_curve_markowitz(prices_train, weights_mvp)
    equity_test_mvp = equity_curve_markowitz(prices_test, weights_mvp)

    metrics_train_mvp = simple_metrics(equity_train_mvp)
    metrics_test_mvp = simple_metrics(equity_test_mvp)

    pd.DataFrame([metrics_train_mvp]).to_csv(
        REPORT_TABLE_DIR / "metrics_train_markowitz.csv", index=False
    )
    pd.DataFrame([metrics_test_mvp]).to_csv(
        REPORT_TABLE_DIR / "metrics_test_markowitz.csv", index=False
    )

    plt.figure(figsize=(10, 5))
    equity_train_mvp.plot(label="Train")
    equity_test_mvp.plot(label="Test")
    plt.title("Markowitz Minimum Variance Portfolio")
    plt.ylabel("Portfolio value (start = 1.0)")
    plt.legend()
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "markowitz_min_variance.png")

    print("Train metrics (Markowitz):", metrics_train_mvp)
    print("Test metrics  (Markowitz):", metrics_test_mvp)


if __name__ == "__main__":
    main()
