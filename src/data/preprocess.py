# src/data/preprocess.py
# Simple preprocessing helpers: cleaning, returns, train/test split.

import pandas as pd

from src.config import TEST_RATIO


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing values.
    return prices.dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # Daily percentage returns from price data.
    returns = prices.pct_change().dropna()
    return returns


def split_train_test(prices: pd.DataFrame):
    # train/test split based on TEST_RATIO.
    split_idx = int(len(prices) * (1 - TEST_RATIO))
    prices_train = prices.iloc[:split_idx]
    prices_test = prices.iloc[split_idx:]
    return prices_train, prices_test
