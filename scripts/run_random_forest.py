# scripts/run_random_forest.py
"""
Train and evaluate a Random Forest supervised strategy.

Steps:
- Train one RF classifier per asset (with ANOVA + GridSearchCV)
- Backtest the RF-based portfolio on the test period
- Compare RF to Equal-Weight and Markowitz MVP
"""

import sys
from pathlib import Path

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import REPORT_FIG_DIR, REPORT_TABLE_DIR, TICKERS
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test, compute_returns
from src.features.technical_indicators import compute_technical_features
from src.baselines.equal_weight import equity_curve_equal_weight
from src.baselines.markowitz import (
    compute_min_variance_weights,
    equity_curve_markowitz,
)
from src.evaluation.metrics import simple_metrics
from src.models.random_forest import (
    train_random_forest_models,
    backtest_rf_portfolio,
)


def main() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load prices and split
    prices = load_prices()
    prices_train, prices_test = split_train_test(prices)
    train_end_date = prices_train.index[-1]

    # 2) Compute technical features on full period
    features = compute_technical_features(prices)

    # 3) Train Random Forest models per asset (with ANOVA + GridSearchCV)
    models, meta = train_random_forest_models(
        prices=prices,
        features=features,
        tickers=TICKERS,
        train_end_date=train_end_date,
        n_estimators=200,
        max_depth=None,
        random_state=42,
        top_k_features=5,
        use_grid_search=True,
    )

    print("--- Feature Selection (ANOVA) Results ---")
    for ticker in TICKERS:
        print(f"{ticker}: {meta[ticker]['selected_features']}")

    print("\n--- Random Forest best hyperparameters (GridSearchCV) ---")
    for ticker in TICKERS:
        print(f"{ticker}: {meta[ticker]['best_params']}")

    # 4) Backtest RF portfolio on test period
    equity_rf = backtest_rf_portfolio(models, meta)

    # 5) Build baselines on the same test period index
    equity_eq_full = equity_curve_equal_weight(prices_test)
    equity_eq = equity_eq_full.loc[equity_rf.index]

    returns_train = compute_returns(prices_train)
    weights_mvp = compute_min_variance_weights(returns_train)
    equity_marko_full = equity_curve_markowitz(prices_test, weights_mvp)
    equity_marko = equity_marko_full.loc[equity_rf.index]

    # 6) Compute metrics for the three strategies
    metrics_eq = simple_metrics(equity_eq)
    metrics_marko = simple_metrics(equity_marko)
    metrics_rf = simple_metrics(equity_rf)

    metrics_df = pd.DataFrame(
        [metrics_eq, metrics_marko, metrics_rf],
        index=["baseline_equal_weight", "markowitz_mvp", "random_forest"],
    )

    metrics_path = REPORT_TABLE_DIR / "metrics_test_random_forest_vs_baselines.csv"
    metrics_df.to_csv(metrics_path)

    # 7) Plot equity curves
    plt.figure(figsize=(10, 5))
    equity_eq.plot(label="Equal-Weight Baseline")
    equity_marko.plot(label="Markowitz MVP")
    equity_rf.plot(label="Random Forest Strategy")
    plt.title("Test Equity Curve: Random Forest vs Baselines")
    plt.ylabel("Portfolio value (start = 1.0)")
    plt.grid()
    plt.legend()
    fig_path = REPORT_FIG_DIR / "equity_random_forest_vs_baselines_test.png"
    plt.savefig(fig_path)

    print("\n Random Forest backtest completed.")
    print("Metrics:\n", metrics_df)
    print("Metrics table:", metrics_path)
    print("Figure:", fig_path)


if __name__ == "__main__":
    main()
