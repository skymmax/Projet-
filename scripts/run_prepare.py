# scripts/run_prepare.py
# build technical features and basic eda figures

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# make sure we can import from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, REPORT_FIG_DIR
from src.data.load_data import load_prices
from src.data.preprocess import compute_returns
from src.features.technical_indicators import compute_technical_features


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- load clean prices ---
    prices = load_prices()

    # --- compute technical features (indicators) ---
    features = compute_technical_features(prices)

    out_path = PROCESSED_DIR / "features_technical.csv"
    features.to_csv(out_path)

    # --- basic eda: returns and correlations ---
    returns = compute_returns(prices)

    # prices normalised
    plt.figure(figsize=(10, 6))
    (prices / prices.iloc[0]).plot()
    plt.title("normalized prices")
    plt.ylabel("normalized value (base = 1)")
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "prices_normalized.png")

    # returns distribution
    plt.figure(figsize=(10, 6))
    returns.plot(kind="hist", bins=50, alpha=0.7)
    plt.title("distribution of daily returns")
    plt.xlabel("daily return")
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "returns_distribution.png")

    # correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
    plt.title("correlation matrix of returns")
    plt.savefig(REPORT_FIG_DIR / "correlation_heatmap.png")

    # rolling volatility
    rolling_vol = returns.rolling(20).std()
    plt.figure(figsize=(10, 6))
    rolling_vol.plot()
    plt.title("rolling volatility (20 days)")
    plt.ylabel("volatility")
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "rolling_volatility.png")

    print("--- prepare step finished ---")
    print("features saved to:", out_path)
    print("eda figures saved in:", REPORT_FIG_DIR)


if __name__ == "__main__":
    main()
