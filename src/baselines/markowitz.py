# src/baselines/markowitz.py
"""
Markowitz Minimum Variance Portfolio (MVP).

The MVP chooses portfolio weights that minimize variance
subject to the constraint that all weights sum to 1.
"""

import numpy as np
import pandas as pd


def compute_min_variance_weights(returns: pd.DataFrame) -> np.ndarray:
 
    #Compute the Minimum Variance Portfolio
   
    # Covariance matrix of returns
    cov = returns.cov().values
    ones = np.ones(cov.shape[0])

    # Inverse covariance matrix
    inv_cov = np.linalg.inv(cov)

    # Apply MVP closed-form formula
    raw_weights = inv_cov @ ones
    weights = raw_weights / raw_weights.sum()

    return weights


def equity_curve_markowitz(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
   
    #Compute the equity curve of a Markowitz Minimum Variance portfolio
    
    prices = prices.dropna()

    # Price relatives compared to the first day
    relatives = prices / prices.iloc[0]

    # Weighted average of relatives using the MVP weights
    equity = (relatives * weights).sum(axis=1)
    equity.name = "equity_markowitz"

    return equity
