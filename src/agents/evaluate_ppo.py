# src/agents/evaluate_ppo.py
"""
Evaluate the trained PPO agent on the test set and compare it
to the Equal-Weight Buy & Hold baseline.

Outputs
- metrics_test_ppo_vs_baseline.csv
- equity_ppo_vs_baseline_test.png
- ppo_weights_test.csv
- ppo_weights_allocation_test.png
"""

import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import REPORT_FIG_DIR, REPORT_TABLE_DIR, PPO_MODEL_PATH, TICKERS
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test
from src.features.technical_indicators import compute_technical_features
from src.baselines.equal_weight import equity_curve_equal_weight
from src.evaluation.metrics import simple_metrics
from src.env.portfolio_env import PortfolioEnv


def make_test_env(prices_test: pd.DataFrame, features_test: pd.DataFrame):
   
    #Return a function that creates a PortfolioEnv for the test period.

   # Used by DummyVecEnv to evaluate the PPO agent.
    

    def _init():
        prices_aligned = prices_test.loc[features_test.index]
        env = PortfolioEnv(
            prices=prices_aligned,
            features=features_test,
            initial_capital=1.0,
            transaction_cost=0.001,
        )
        return env

    return _init


def evaluate_ppo() -> None:
    # Run one test episode with PPO and compare to Equal-Weight.

    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load data and split
    prices = load_prices()
    _, prices_test = split_train_test(prices)

    # 2) Compute technical features on the test set
    features_test = compute_technical_features(prices_test)

    # Align prices to features index (drop warmup period)
    prices_test_aligned = prices_test.loc[features_test.index]

    # 3) Create test environment (vectorized) 
    test_env_fn = make_test_env(prices_test_aligned, features_test)
    vec_env = DummyVecEnv([test_env_fn])

    #  4) Load PPO model
    model = PPO.load(str(PPO_MODEL_PATH))

    # 5) Roll out one full episode in deterministic mode
    obs = vec_env.reset()
    done = False

    portfolio_values = []
    dates = []
    weights_list = []  # store allocation over time

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        done = dones[0]
        info = infos[0]

        portfolio_values.append(info["portfolio_value"])
        dates.append(info["date"])
        if "weights" in info:
            weights_list.append(info["weights"])

    equity_ppo = pd.Series(
        portfolio_values,
        index=pd.to_datetime(dates),
        name="equity_ppo",
    )

    #  6 Save weights over time (if available)
    if weights_list:
        weights_df = pd.DataFrame(
            weights_list,
            index=pd.to_datetime(dates),
            columns=TICKERS,
        )
        weights_path = REPORT_TABLE_DIR / "ppo_weights_test.csv"
        weights_df.to_csv(weights_path)

    # 7 Baseline equity on the same dates
    equity_baseline_full = equity_curve_equal_weight(prices_test_aligned)
    equity_baseline = equity_baseline_full.loc[equity_ppo.index]
    equity_baseline.name = "equity_baseline_equal_weight"

    # 8) Compute metrics
    metrics_baseline = simple_metrics(equity_baseline)
    metrics_ppo = simple_metrics(equity_ppo)

    metrics_df = pd.DataFrame(
        [metrics_baseline, metrics_ppo],
        index=["baseline_equal_weight", "ppo_rl"],
    )

    # 9) Save metrics table
    metrics_path = REPORT_TABLE_DIR / "metrics_test_ppo_vs_baseline.csv"
    metrics_df.to_csv(metrics_path)

    # 10) Plot equity curves
    plt.figure(figsize=(10, 5))
    equity_baseline.plot(label="Baseline Equal-Weight")
    equity_ppo.plot(label="PPO RL Agent")
    plt.title("Test Equity Curve: PPO vs Equal-Weight Baseline")
    plt.ylabel("Portfolio value (start = 1.0)")
    plt.grid()
    plt.legend()
    fig_path = REPORT_FIG_DIR / "equity_ppo_vs_baseline_test.png"
    plt.savefig(fig_path)

    # 11) Plot allocation over time (stacked area)
    if weights_list:
        plt.figure(figsize=(10, 5))
        weights_df.plot.area()
        plt.title("PPO Portfolio Weights over Time (Test Set)")
        plt.ylabel("Weight")
        plt.ylim(0.0, 1.0)
        plt.grid(axis="y", alpha=0.3)
        plt.legend(title="Ticker", loc="upper left")
        alloc_fig_path = REPORT_FIG_DIR / "ppo_weights_allocation_test.png"
        plt.savefig(alloc_fig_path)
    else:
        alloc_fig_path = None

    # 12) Short summary in the console
    print("PPO evaluation finished")
    print("Metrics file:", metrics_path)
    print("Equity figure:", fig_path)
    if alloc_fig_path is not None:
        print("Allocation figure:", alloc_fig_path)


def main():
    evaluate_ppo()


if __name__ == "__main__":
    main()
