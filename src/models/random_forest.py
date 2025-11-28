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

    For each test date:
        - each model predicts P(UP) for its own asset
        - weights are proportional to P(UP) (if all zero -> equal weight)
        - portfolio return is the weighted sum of next-day returns

    Returns:
        equity: pd.Series of portfolio value over time on the common test index.
    """
    tickers = list(models.keys())

    # Common index across all tickers' test sets
    common_index = None
    for ticker in tickers:
        idx = meta[ticker]["X_test"].index
        common_index = idx if common_index is None else common_index.intersection(idx)

    common_index = common_index.sort_values()

    equity_values = []
    current_value = initial_capital

    for date in common_index:
        prob_up_list = []
        ret_next_list = []

        for ticker in tickers:
            X_test = meta[ticker]["X_test"]
            ret_next_test = meta[ticker]["ret_next_test"]

            x_row = X_test.loc[date].values.reshape(1, -1)
            proba = models[ticker].predict_proba(x_row)[0, 1]  # P(class=1, UP)

            prob_up_list.append(proba)
            ret_next_list.append(ret_next_test.loc[date])

        prob_up = np.array(prob_up_list, dtype=float)
        returns_next = np.array(ret_next_list, dtype=float)

        # Convert probabilities to weights (long-only)
        if prob_up.sum() <= 0:
            weights = np.full_like(prob_up, 1.0 / len(prob_up))
        else:
            weights = prob_up / prob_up.sum()

        # Portfolio return for this step
        portfolio_ret = float((weights * returns_next).sum())

        current_value *= (1.0 + portfolio_ret)
        equity_values.append(current_value)

    equity = pd.Series(equity_values, index=common_index, name="equity_random_forest")
    return equity
