# src/data/load_data.py
# Download the S&P500 dataset from Kaggle and build a prices DataFrame.

from pathlib import Path

import kagglehub
import pandas as pd

from src.config import RAW_DIR, TICKERS, START_DATE, END_DATE
from src.data.preprocess import clean_prices


def download_sp500_csv() -> Path:
    # Download for the Kaggle S&P500 dataset.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download("camnugent/sandp500"))
    csv_path = dataset_path / "all_stocks_5yr.csv"
    return csv_path


def load_raw_sp500() -> pd.DataFrame:
    # Load raw Kaggle CSV into a DataFrame.
    csv_path = download_sp500_csv()
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_prices(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Keep only selected tickers and build a wide prices table.
    df = df_raw.loc[df_raw["Name"].isin(TICKERS), ["date", "Name", "close"]].copy()
    df.rename(columns={"Name": "ticker", "close": "price"}, inplace=True)
    prices = df.pivot(index="date", columns="ticker", values="price").sort_index()
    return prices


def load_prices() -> pd.DataFrame:
    df_raw = load_raw_sp500()
    prices = build_prices(df_raw)
    prices = clean_prices(prices)
    prices = prices.sort_index()
    prices = prices.loc[START_DATE:END_DATE]
    return prices
