# src/data/load_data.py

import pandas as pd
from pathlib import Path
import kagglehub

from src.config import RAW_DIR, TICKERS
from src.data.preprocess import clean_prices
from src.config import START_DATE, END_DATE



def download_sp500_csv() -> Path:
    """
    Download the S&P500 5-year dataset from Kaggle using kagglehub.
    Returns the path to the CSV file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # This downloads (or reuses cached) Kaggle dataset
    dataset_path = Path(kagglehub.dataset_download("camnugent/sandp500"))
    csv_path = dataset_path / "all_stocks_5yr.csv"
    return csv_path


def load_raw_sp500() -> pd.DataFrame:
    """
    Load the raw Kaggle CSV into a DataFrame.
    """
    csv_path = download_sp500_csv()
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_prices(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    From the raw Kaggle dataframe, keep only our tickers and build
    a wide dataframe: index = date, columns = tickers, values = close price.
    """
    df = df_raw.loc[df_raw["Name"].isin(TICKERS), ["date", "Name", "close"]].copy()
    df.rename(columns={"Name": "ticker", "close": "price"}, inplace=True)

    prices = df.pivot(index="date", columns="ticker", values="price").sort_index()
    return prices


def load_prices() -> pd.DataFrame:
    """
    Main function used everywhere in the project:
    returns a clean prices dataframe.
    """
    df_raw = load_raw_sp500()
    prices = build_prices(df_raw)
    prices = clean_prices(prices)
    
    prices = prices.sort_index()

    prices = prices.loc[START_DATE:END_DATE]
    
    return prices
