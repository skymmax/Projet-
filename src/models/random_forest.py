# src/models/random_forest.py
# random forest models for supervised next-day direction prediction

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.features.feature_selection import anova_feature_selection


def build_rf_data_for_ticker(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    ticker: str,
    train_end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None:
    # build supervised dataset for one ticker
    # X: technical features at date t
    # y: 1 if return_{t+1} > 0, else 0

    cols = [c for c in features.columns if c.startswith(f"{ticker}_")]
    if not cols:
        # no features for this ticker, skip it
        return None

    X_all = features[cols].copy()

    # next-day returns as target
    returns = prices[ticker].pct_change()
    ret_next = returns.shift(-1)

    df = X_all.copy()
    df["ret_next"] = ret_next
    df = df.dropna()

    y_all = (df["ret_next"] > 0).astype(int)
    X_all = df[cols]

    # train / test split based on date
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
    top_k_features: int = 5,
    use_grid_search: bool = True,
) -> Tuple[Dict[str, RandomForestClassifier], Dict[str, dict]]:
    # train one random forest per ticker to predict next-day up/down

    models: Dict[str, RandomForestClassifier] = {}
    meta: Dict[str, dict] = {}

    for ticker in tickers:
        result = build_rf_data_for_ticker(
            prices, features, ticker, train_end_date
        )

        # if no data for this ticker, move on
        if result is None:
            continue

        X_train_full, y_train, X_test_full, ret_next_test = result

        # anova feature selection
        selected_features = anova_feature_selection(
            X_train_full,
            y_train,
            top_k=top_k_features,
        )

        X_train = X_train_full[selected_features]
        X_test = X_test_full[selected_features]

        best_params = None

        # optional grid search for basic hyperparameters
        if use_grid_search:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            }

            base_rf = RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
            )

            grid = GridSearchCV(
                estimator=base_rf,
                param_grid=param_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )

            grid.fit(X_train, y_train)
            clf = grid.best_estimator_
            best_params = grid.best_params_
        else:
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
            "selected_features": selected_features,
            "train_size": len(X_train),
            "best_params": best_params,
        }

    return models, meta


def backtest_rf_portfolio(
    models: Dict[str, RandomForestClassifier],
    meta: Dict[str, dict],
    initial_capital: float = 1.0,
) -> pd.Series:
    # backtest a long-only rf-based portfolio on the test period

    tickers = list(models.keys())

    # common index across all tickers test sets
    common_index = None
    for ticker in tickers:
        idx = meta[ticker]["X_test"].index
        common_index = idx if common_index is None else common_index.intersection(idx)

    common_index = common_index.sort_values()
    n_dates = len(common_index)
    n_assets = len(tickers)

    # matrices prob_up[date, asset] and returns_next[date, asset]
    prob_matrix = np.zeros((n_dates, n_assets), dtype=float)
    ret_matrix = np.zeros((n_dates, n_assets), dtype=float)

    for j, ticker in enumerate(tickers):
        X_test_full = meta[ticker]["X_test"].loc[common_index]
        ret_next_full = meta[ticker]["ret_next_test"].loc[common_index]

        proba_full = models[ticker].predict_proba(X_test_full)[:, 1]

        prob_matrix[:, j] = proba_full
        ret_matrix[:, j] = ret_next_full.values

    # iterate over time to compute equity curve
    equity_values = []
    current_value = initial_capital

    for i, date in enumerate(common_index):
        probs = prob_matrix[i, :]
        rets = ret_matrix[i, :]

        # convert probabilities to weights
        if probs.sum() <= 0:
            weights = np.full_like(probs, 1.0 / len(probs))
        else:
            weights = probs / probs.sum()

        portfolio_ret = float((weights * rets).sum())
        current_value *= (1.0 + portfolio_ret)
        equity_values.append(current_value)

    equity = pd.Series(
        equity_values,
        index=common_index,
        name="equity_random_forest",
    )
    return equity
