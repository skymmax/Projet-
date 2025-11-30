Portfolio Allocation with Machine Learning & Reinforcement Learning

Dynamic Portfolio Management with Random Forests and PPO (Stable-Baselines3)
Master ESILV – Machine Learning Project

Introduction

This project explores dynamic portfolio management using both classical financial models and machine learning approaches, including supervised learning (Random Forest) and reinforcement learning (PPO).

The goal is to build a system that:

loads and processes stock price data,

computes technical indicators,

trains multiple allocation strategies,

evaluates them with financial metrics,

and compares all results in a unified framework.

We study four strategies:

Equal-Weight Buy & Hold

Markowitz Minimum Variance Portfolio

Random Forest (Supervised ML)

PPO Agent (Reinforcement Learning)

Each method uses the same underlying dataset and evaluation metrics, making comparisons consistent.

Project Structure

project/
│
├── run_all.py                    # runs full pipeline (recommended)
│
├── scripts/
│   ├── run_prepare.py            # build features + generate EDA figures
│   ├── run_baselines.py          # equal-weight + markowitz
│   ├── run_random_forest.py      # random forest supervised strategy
│   └── run_ppo.py                # train + evaluate PPO agent
│
├── src/
│   ├── config.py                 # central parameters & global paths
│   │
│   ├── data/
│   │   ├── load_data.py          # download & load the S&P500 dataset
│   │   └── preprocess.py         # clean, returns, train/test split
│   │
│   ├── features/
│   │   ├── technical_indicators.py   # MA20, MA60, VOL20, returns, etc.
│   │   └── feature_selection.py       # ANOVA F-test (RF)
│   │
│   ├── baselines/
│   │   ├── equal_weight.py       # buy & hold equal-weights
│   │   └── markowitz.py          # minimum variance portfolio
│   │
│   ├── models/
│   │   └── random_forest.py      # train RFs + backtest strategy
│   │
│   ├── env/
│   │   └── portfolio_env.py      # PPO trading environment
│   │
│   ├── agents/
│   │   ├── train_ppo.py          # PPO training
│   │   └── evaluate_ppo.py       # PPO evaluation + plots
│   │
│   └── evaluation/
│       └── metrics.py            # sharpe, drawdown, calmar, etc.
│
├── data/
│   ├── raw/                      # auto-downloaded Kaggle dataset
│   └── processed/                # prepared features
│
├── reports/
│   ├── figures/                  # all generated plots
│   └── tables/                   # all evaluation metrics
│
└── models/
    └── ppo_portfolio.zip         # saved PPO agent

Installation
1. Clone the repository

    git clone <repository_url>
    cd project/

2. Create a virtual environment
    python -m venv .venv
    source .venv/bin/activate        # macOS / Linux
    .venv\Scripts\activate           # windows

3. Install dependencies
    pip install -r requirements.txt

Running the Full Pipeline

Run everything in one command:
    python run_all.py

This executes:

data preparation + EDA

baselines (EW + Markowitz)

Random Forest strategy

PPO training

PPO evaluation

saving all figures & metrics

Run Steps Individually
1) Prepare Data + Features + EDA
    python scripts/run_prepare.py

Outputs:

features_technical.csv

prices_normalized.png

returns_distribution.png

correlation_heatmap.png

rolling_volatility.png

2) Baselines (Equal-Weight & Markowitz)
    python scripts/run_baselines.py

Outputs:

baseline_equal_weight.png

markowitz_min_variance.png

metrics_train_baseline.csv

metrics_test_baseline.csv

metrics_train_markowitz.csv

metrics_test_markowitz.csv

3) Random Forest Strategy
    python scripts/run_random_forest.py

Features:

one classifier per ticker

ANOVA F-test feature selection

optional GridSearchCV hyperparameter tuning

Outputs:

equity_random_forest_vs_baselines_test.png

metrics_test_random_forest_vs_baselines.csv

4) PPO Agent (Train + Evaluate)
    python scripts/run_ppo.py

Outputs:

equity_ppo_vs_baseline_test.png

ppo_weights_allocation_test.png

ppo_weights_test.csv

metrics_test_ppo_vs_baseline.csv

saved model → models/ppo_portfolio.zip

How the PPO Environment Works

The agent interacts with a custom PortfolioEnv (Gymnasium).

Observations

At each step:

price

return

MA20, MA60

VOL20
for each ticker, flattened into one vector.

Action

A vector of raw weights → normalized to a valid portfolio:

weights ≥ 0

sum(weights) = 1

per-asset cap (default 40%)

Reward
reward = log(1 + net_return) + drawdown_penalty * drawdown


The reward encourages:

higher returns

low drawdown

smooth allocation changes

Info Returned

Useful for evaluation and plotting:

portfolio value

turnover

returns

drawdown

next date

current weights

Metrics

All strategies use the same evaluation metrics:

Metric	            Description
Total Return	    Final growth factor
Daily Volatility	Variance of returns
Sharpe Ratio	    Mean return / volatility
Max Drawdown	    Worst historical decline
Calmar Ratio	    Annual return / max drawdown

Results saved in:

reports/tables/

Generated Outputs

All plots are stored in:

reports/figures/


Examples:

normalized prices

returns distribution

rolling volatility

heatmap

equal-weight curve

markowitz curve

random forest vs baselines

PPO vs baseline

PPO allocation weights

Metrics (CSV) stored in:

reports/tables/

Technical Choices

Kaggle dataset automatically downloaded

technical indicators: MA20, MA60, RET, VOL20

ANOVA for RF feature selection

GridSearchCV for RF tuning

PPO (SB3) with custom financial reward

Gymnasium environment tailored to multi-asset allocation

risk-aware reward including drawdown penalties

clean modular architecture

Conclusion

This project provides a complete, modular and educational framework for:

exploring portfolio optimization

combining machine learning with quantitative finance

comparing classical & modern approaches

studying agent behavior and risk control

It is fully reproducible and designed to be extended (new assets, new models, new reward functions).