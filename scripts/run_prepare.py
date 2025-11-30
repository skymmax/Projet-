# scripts/run_prepare.py
# Build and save the technical feature dataset used by all models.
# Outputs: data/processed/features_technical.csv

import sys
from pathlib import Path

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR
from src.data.load_data import load_prices
from src.features.technical_indicators import compute_technical_features


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load clean prices ---
    prices = load_prices()
    print("--- Prices loaded ---")
    print("Shape:", prices.shape)

    # --- Compute technical features (indicators) ---
    features = compute_technical_features(prices)
    print("--- Technical features computed ---")
    print("Shape:", features.shape)
    print("First columns:", list(features.columns)[:10])

    # --- Save to disk ---
    out_path = PROCESSED_DIR / "features_technical.csv"
    features.to_csv(out_path)

    print("--- Prepare step finished ---")
    print("Saved technical features to:", out_path)


if __name__ == "__main__":
    main()
