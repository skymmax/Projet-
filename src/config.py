# src/config.py

# Global configuration for the project.

# NOTE:
# you only need to edit the block "USER SETTINGS" below.


from pathlib import Path

# ----------------------------------------------------
# USER SETTINGS (EDIT IF NEEDED)
# ----------------------------------------------------

# List of tickers used in the portfolio
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "GS"]

# Global time period for the project
START_DATE = "2013-01-01"   # start of historical data
END_DATE = "2018-03-01"     # end of historical data

# Train / test split (ratio of the TOTAL period reserved for test)
TEST_RATIO = 0.3

# Global random seed (for reproducibility)
SEED = 42

# ----------------------------------------------------
# PATHS (DO NOT EDIT)
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR = DATA_DIR / "raw"
REPORT_FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

MODELS_DIR = PROJECT_ROOT / "models"
PPO_MODEL_PATH = MODELS_DIR / "ppo_portfolio"
