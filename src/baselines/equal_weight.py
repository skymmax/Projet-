# src/baselines/equal_weight.py

import numpy as np
import pandas as pd


def equity_curve_equal_weight(prices_df: pd.DataFrame) -> pd.Series:
    """
    Compute the equity curve of a simple Equal-Weight Buy & Hold strategy.

    Steps:
    - invest 1/N in each asset on day 0
    - no rebalancing
    - portfolio value evolves according to price relatives

    Returns:
        pd.Series indexed by date, representing portfolio value over time.
    """
    prices_df = prices_df.dropna()

    n = prices_df.shape[1]  # number of assets
    if n == 0:
        raise ValueError("Prices DataFrame contains no assets.")

    # Equal weights 1/N
    weights = np.full(n, 1.0 / n)

    # Price relatives compared to initial day
    relatives = prices_df / prices_df.iloc[0]

    # Portfolio equity curve
    equity_curve = (relatives * weights).sum(axis=1)
    equity_curve.name = "equity_equal_weight"

    return equity_curve
