# src/models/random_forest.py

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def build_rf_data_for_ticker(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    ticker: str,
    train_end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build supervised dataset for one ticker:
    - X: technical features at date t
    - y: binary label = 1 if return_{t+1} > 0, else 0

    We split X, y into train / test based on train_end_date.
    """
    # Columns corresponding to this ticker
    cols = [c for c in features.columns if c.startswith(f"{ticker}_")]
    if not cols:
        raise ValueError(f"No feature columns found for ticker {ticker}.")

    X_all = features[cols].copy()

    # Next-day returns as target
    returns = prices[ticker].pct_change()
    ret_next = returns.shift(-1)

    df = X_all.copy()
    df["ret_next"] = ret_next
    df = df.dropna()

    y_all = (df["ret_next"] > 0).astype(int)
    X_all = df[cols]

    # Train / test split based on date
    train_mask = df.index <= train_end_date
    X_train = X_all.loc[train_mask]
    y_train = y_all.loc[train_mask]

    X_test = X_all.loc[~train_mask]
    ret_next_test = df.loc[~train_mask, "ret_next"]

    return X_train, y_train, X_test, ret_next_test


def train_random_forest_models(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    tickers: list[str],
    train_end_date: pd.Timestamp,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Tuple[Dict[str, RandomForestClassifier], Dict[str, dict]]:
    """
    Train one RandomForestClassifier per ticker to predict next-day UP/DOWN.

    Returns:
        models: dict[ticker] -> trained RandomForestClassifier
        meta:   dict[ticker] -> {
                    "X_test": X_test,
                    "ret_next_test": ret_next_test,
                    "train_size": len(X_train),
                    "feature_cols": feature column names,
                }
    """
    models: Dict[str, RandomForestClassifier] = {}
    meta: Dict[str, dict] = {}

    for ticker in tickers:
        X_train, y_train, X_test, ret_next_test = build_rf_data_for_ticker(
            prices, features, ticker, train_end_date
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        models[ticker] = clf
        meta[ticker] = {
            "X_test": X_test,
            "ret_next_test": ret_next_test,
            "train_size": len(X_train),
            "feature_cols": list(X_train.columns),
        }

    return models, meta


def backtest_rf_portfolio(
    models: Dict[str, RandomForestClassifier],
    meta: Dict[str, dict],
    initial_capital: float = 1.0,
) -> pd.Series:
    """
    Backtest a long-only RF-based portfolio on the test period.

    Vectorized version:
        - for each ticker, compute P(UP) for all test dates at once
        - build a matrix of probabilities and a matrix of next-day returns
        - iterate once over time to compute the portfolio equity.

    Returns:
        equity: pd.Series of portfolio value over time on the common test index.
    """
    tickers = list(models.keys())

    # 1) Common index across all tickers' test sets
    common_index = None
    for ticker in tickers:
        idx = meta[ticker]["X_test"].index
        common_index = idx if common_index is None else common_index.intersection(idx)

    common_index = common_index.sort_values()
    n_dates = len(common_index)
    n_assets = len(tickers)

    # 2) Matrices prob_up[date, asset] and returns_next[date, asset]
    prob_matrix = np.zeros((n_dates, n_assets), dtype=float)
    ret_matrix = np.zeros((n_dates, n_assets), dtype=float)

    for j, ticker in enumerate(tickers):
        X_test_full = meta[ticker]["X_test"].loc[common_index]
        ret_next_full = meta[ticker]["ret_next_test"].loc[common_index]

        # Predict P(UP) for all dates at once (no loop on dates)
        proba_full = models[ticker].predict_proba(X_test_full)[:, 1]

        prob_matrix[:, j] = proba_full
        ret_matrix[:, j] = ret_next_full.values

    # 3) Iterate over time to compute equity curve
    equity_values = []
    current_value = initial_capital

    for i, date in enumerate(common_index):
        probs = prob_matrix[i, :]
        rets = ret_matrix[i, :]

        # Convert probabilities to weights
        if probs.sum() <= 0:
            weights = np.full_like(probs, 1.0 / len(probs))
        else:
            weights = probs / probs.sum()

        # Portfolio return at this step
        portfolio_ret = float((weights * rets).sum())
        current_value *= (1.0 + portfolio_ret)
        equity_values.append(current_value)

    equity = pd.Series(equity_values, index=common_index, name="equity_random_forest")
    return equity
