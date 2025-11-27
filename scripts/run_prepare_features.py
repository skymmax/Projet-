# scripts/run_prepare_features.py

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.load_data import load_prices
from src.features.technical_indicators import compute_technical_features


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load clean prices
    prices = load_prices()
    print("Prices loaded")
    print("Shape:", prices.shape)

    # 2) Compute technical features
    features = compute_technical_features(prices)
    print("Technical features computed")
    print("Shape:", features.shape)
    print("Columns example:", list(features.columns)[:10])

    # 3) Save to disk
    out_path = PROCESSED_DIR / "features_technical.csv"
    features.to_csv(out_path)

    print("\nSaved technical features to:", out_path)


if __name__ == "__main__":
    main()
