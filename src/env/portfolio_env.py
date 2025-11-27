# src/env/portfolio_env.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Gymnasium-compatible environment for portfolio allocation.

    - Observation: feature vector for the current date (e.g. technical indicators)
    - Action: portfolio weights over assets (continuous, sum to 1)
    - Reward: portfolio daily return (optionally minus transaction costs)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        initial_capital: float = 1.0,
        transaction_cost: float = 0.0,
    ):
        super().__init__()

        # Sort by date to be safe
        self.prices = prices.sort_index()

        # If external features are provided, align on common dates
        if features is not None:
            features = features.sort_index()
            common_idx = self.prices.index.intersection(features.index)
            self.prices = self.prices.loc[common_idx]
            self.features = features.loc[common_idx]
        else:
            # Fallback: simple default features based on returns
            self.features = self._build_default_features(self.prices)

        # Compute daily returns (fill first day with zeros)
        self.returns = self.prices.pct_change().fillna(0.0)

        self.dates = self.features.index
        self.n_steps = len(self.dates)
        self.n_assets = self.prices.shape[1]
        self.n_features = self.features.shape[1]

        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        # Observation: feature vector for one date
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # Action: weights per asset (continuous [0, 1])
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # Internal state
        self._current_step: int = 0
        self._portfolio_value: float = self.initial_capital
        self._weights = np.full(self.n_assets, 1.0 / self.n_assets)

    # ---------- Helpers ----------

    def _build_default_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Default fallback features: simply use daily returns of all assets.
        """
        returns = prices.pct_change().fillna(0.0)
        return returns

    def _get_observation(self) -> np.ndarray:
        """
        Return the feature vector at the current step.
        """
        row = self.features.iloc[self._current_step].astype(np.float32).values
        return row

    # ---------- Gymnasium API ----------

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._portfolio_value = self.initial_capital
        self._weights = np.full(self.n_assets, 1.0 / self.n_assets)

        obs = self._get_observation()
        info = {
            "portfolio_value": self._portfolio_value,
            "step": self._current_step,
            "date": self.dates[self._current_step],
        }
        return obs, info

    def step(self, action: np.ndarray):
        # Clip action into [0, 1] and renormalize to sum to 1
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        if action.sum() <= 0:
            new_weights = np.full(self.n_assets, 1.0 / self.n_assets)
        else:
            new_weights = action / action.sum()

        # Transaction costs based on turnover (L1 distance between weight vectors)
        turnover = float(np.abs(new_weights - self._weights).sum())
        cost = self.transaction_cost * turnover

        # Current date and asset returns
        current_date = self.dates[self._current_step]
        asset_returns = self.returns.loc[current_date].values.astype(np.float32)

        # Portfolio return before costs
        portfolio_return = float(np.dot(new_weights, asset_returns))

        # Update portfolio value (net of transaction costs)
        self._portfolio_value *= (1.0 + portfolio_return - cost)

        # Update internal state
        self._weights = new_weights
        self._current_step += 1

        terminated = self._current_step >= (self.n_steps - 1)
        truncated = False

        # Reward: here simply the daily portfolio return
        reward = portfolio_return

        if not terminated:
            obs = self._get_observation()
            info_date = self.dates[self._current_step]
        else:
            # Dummy obs at the end
            obs = np.zeros(self.n_features, dtype=np.float32)
            info_date = self.dates[self._current_step - 1]

        info = {
            "portfolio_value": self._portfolio_value,
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "step": self._current_step,
            "date": info_date,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Step {self._current_step} | "
            f"Value={self._portfolio_value:.4f} | "
            f"Weights={np.round(self._weights, 3)}"
        )
