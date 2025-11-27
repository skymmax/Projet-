# src/baselines/markowitz.py

import numpy as np
import pandas as pd


def compute_min_variance_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the Minimum Variance Portfolio (MVP) weights:
        w = Σ^{-1} * 1 / (1^T * Σ^{-1} * 1)

    Args:
        returns: DataFrame of daily returns for each asset.

    Returns:
        np.ndarray of weights summing to 1.
    """
    cov = returns.cov().values  # covariance matrix
    ones = np.ones(cov.shape[0])

    # inverse covariance
    inv_cov = np.linalg.inv(cov)

    # MVP formula
    raw_weights = inv_cov @ ones
    weights = raw_weights / raw_weights.sum()

    return weights


def equity_curve_markowitz(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Compute the equity curve of a Markowitz Minimum Variance portfolio.

    Args:
        prices: DataFrame of asset prices
        weights: optimal weights from the MVP

    Returns:
        pd.Series with the portfolio value over time.
    """
    prices = prices.dropna()
    relatives = prices / prices.iloc[0]
    equity = (relatives * weights).sum(axis=1)
    equity.name = "equity_markowitz"
    return equity
