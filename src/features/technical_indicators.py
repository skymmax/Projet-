# src/features/technical_indicators.py
# basic technical indicators used as features

import pandas as pd

from src.data.preprocess import compute_returns


def add_moving_averages(prices: pd.DataFrame, windows=(20, 60)) -> pd.DataFrame:
    # simple moving averages for each asset
    df = prices.copy()
    for w in windows:
        df[f"MA{w}"] = prices.rolling(window=w).mean()
    return df


def add_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    # rolling volatility of returns
    vol = returns.rolling(window=window).std()
    vol.columns = [f"{c}_VOL{window}" for c in vol.columns]
    return vol


def compute_technical_features(
    prices: pd.DataFrame,
    ma_windows=(20, 60),
    vol_window: int = 20,
) -> pd.DataFrame:
    # compute basic features for each asset:
    # price, daily return, moving averages, rolling volatility
    returns = compute_returns(prices)

    feature_frames = []

    for ticker in prices.columns:
        p = prices[ticker]
        r = returns[ticker]

        df_t = pd.DataFrame(index=prices.index)
        df_t[f"{ticker}_PRICE"] = p
        df_t[f"{ticker}_RET"] = r

        for w in ma_windows:
            df_t[f"{ticker}_MA{w}"] = p.rolling(window=w).mean()

        df_t[f"{ticker}_VOL{vol_window}"] = r.rolling(window=vol_window).std()

        feature_frames.append(df_t)

    features = pd.concat(feature_frames, axis=1)
    features = features.dropna()

    return features
