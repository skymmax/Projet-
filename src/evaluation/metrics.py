# src/evaluation/metrics.py

import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Compute the maximum drawdown of a portfolio equity curve.

    Max drawdown = min_t ( equity_t / max_{s<=t}(equity_s) - 1 )

    Returns:
        A negative float (e.g. -0.25 for -25% max drawdown).
    """
    # Cumulative maximum of equity
    cumulative_max = equity.cummax()

    # Drawdown at each time step
    drawdowns = equity / cumulative_max - 1.0

    # Minimum (most negative) drawdown
    max_drawdown = float(drawdowns.min())

    return max_drawdown


def simple_metrics(equity: pd.Series) -> dict:
    """
    Compute basic financial metrics from a portfolio equity curve:
    - total return
    - daily volatility
    - daily Sharpe ratio (rf = 0)
    - max drawdown
    - Calmar ratio (annualized return / |max drawdown|)
    """
    returns = equity.pct_change().dropna()

    # Total return over the full period
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)

    # Daily volatility and Sharpe
    volatility = float(returns.std())  # daily std
    sharpe = float(0 if volatility == 0 else returns.mean() / volatility)

    # Max drawdown
    max_drawdown = compute_max_drawdown(equity)

    # Calmar ratio: annualized return / |max_drawdown|
    # Approximate 252 trading days per year
    n_days = len(equity)
    if n_days > 1:
        annual_return = float(
            (equity.iloc[-1] / equity.iloc[0]) ** (252 / (n_days - 1)) - 1
        )
    else:
        annual_return = 0.0

    if max_drawdown < 0:
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        # If no drawdown (theoretically), Calmar is not defined -> set to 0
        calmar_ratio = 0.0

    return {
        "total_return": total_return,
        "daily_volatility": volatility,
        "daily_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }
