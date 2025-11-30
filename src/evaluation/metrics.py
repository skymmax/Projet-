# src/evaluation/metrics.py
# simple performance metrics used throughout the project

import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    # max drawdown = min over time of equity / rolling peak - 1
    cumulative_max = equity.cummax()
    drawdowns = equity / cumulative_max - 1.0
    return float(drawdowns.min())


def simple_metrics(equity: pd.Series) -> dict:
    # compute basic portfolio metrics
    returns = equity.pct_change().dropna()

    # total return from start to end
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)

    # daily volatility and sharpe (rf = 0)
    volatility = float(returns.std())
    sharpe = float(0 if volatility == 0 else returns.mean() / volatility)

    # max drawdown
    max_dd = compute_max_drawdown(equity)

    # annual return approximation (252 trading days)
    n_days = len(equity)
    if n_days > 1:
        annual_return = float(
            (equity.iloc[-1] / equity.iloc[0]) ** (252 / (n_days - 1)) - 1
        )
    else:
        annual_return = 0.0

    # calmar ratio = annual return / abs (max drawdown)
    if max_dd < 0:
        calmar = annual_return / abs(max_dd)
    else:
        calmar = 0.0

    return {
        "total_return": total_return,
        "daily_volatility": volatility,
        "daily_sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
    }
