# src/features/technical_indicators.py

import pandas as pd

from src.data.preprocess import compute_returns


def add_moving_averages(prices: pd.DataFrame, windows=(20, 60)) -> pd.DataFrame:
    df = prices.copy()
    for w in windows:
        df[f"MA{w}"] = prices.rolling(window=w).mean()
    return df

def add_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    vol = returns.rolling(window=window).std()
    vol.columns = [f"{c}_VOL{window}" for c in vol.columns]
    return vol



def compute_technical_features(
    prices: pd.DataFrame,
    ma_windows=(20, 60),
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Compute technical features for each asset:
    - price
    - daily return
    - moving averages (MA)
    - rolling volatility of returns

    Returns a wide DataFrame indexed by date, with one column per (asset, feature),
    e.g. AAPL_PRICE, AAPL_RET, AAPL_MA20, AAPL_MA60, AAPL_VOL20, ...
    """
    # Base objects
    returns = compute_returns(prices)

    feature_frames = []

    for ticker in prices.columns:
        p = prices[ticker]
        r = returns[ticker]

        df_t = pd.DataFrame(index=prices.index)
        df_t[f"{ticker}_PRICE"] = p

        # Daily return (aligned with prices index)
        df_t[f"{ticker}_RET"] = r

        # Moving averages
        for w in ma_windows:
            df_t[f"{ticker}_MA{w}"] = p.rolling(window=w).mean()

        # Rolling volatility of returns
        df_t[f"{ticker}_VOL{vol_window}"] = r.rolling(window=vol_window).std()

        feature_frames.append(df_t)

    # Concatenate all tickers horizontally
    features = pd.concat(feature_frames, axis=1)

    # Drop rows with any missing values (due to rolling windows)
    features = features.dropna()

    return features
