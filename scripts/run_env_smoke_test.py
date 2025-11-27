# scripts/run_env_smoke_test.py

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.load_data import load_prices
from src.data.preprocess import split_train_test
from src.features.technical_indicators import compute_technical_features
from src.env.portfolio_env import PortfolioEnv


def main():
    # 1) Load data
    prices = load_prices()
    prices_train, _ = split_train_test(prices)

    # 2) Compute technical features on the train set
    features_train = compute_technical_features(prices_train)

    # Align prices to features index
    prices_train_aligned = prices_train.loc[features_train.index]

    # 3) Create environment
    env = PortfolioEnv(
        prices=prices_train_aligned,
        features=features_train,
        initial_capital=1.0,
        transaction_cost=0.001,  # 0.1% per unit of turnover
    )

    # 4) Run a short random episode
    obs, info = env.reset()
    done = False
    steps = 0

    while not done and steps < 50:
        action = env.action_space.sample()  # random weights
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    print("✅ Smoke test terminé")
    print("Nombre d'étapes simulées :", steps)
    print("Valeur finale du portefeuille :", round(info["portfolio_value"], 4))


if __name__ == "__main__":
    main()
