# src/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

# Finance / dataset
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL"]  # Example tickers

# Global time period for the project
START_DATE = "2013-01-01"   # début de l’historique
END_DATE = "2018-03-01"     # fin de l’historique

# Train / test split (ratio sur la période START–END)
TEST_RATIO = 0.2

SEED = 42

MODELS_DIR = PROJECT_ROOT / "models"
PPO_MODEL_PATH = MODELS_DIR / "ppo_portfolio"