# src/env/portfolio_env.py
# Gymnasium-compatible environment for portfolio allocation.
# The agent chooses portfolio weights, receives reward based on returns,
# and the environment tracks portfolio value and drawdown.

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        initial_capital: float = 1.0,
        transaction_cost: float = 0.0,
        drawdown_penalty: float = 0.1,
        max_weight_per_asset: float = 0.4,
    ):
        super().__init__()

        # Sort by date
        self.prices = prices.sort_index()

        # Align with features if provided
        if features is not None:
            features = features.sort_index()
            common_idx = self.prices.index.intersection(features.index)
            self.prices = self.prices.loc[common_idx]
            self.features = features.loc[common_idx]
        else:
            # Fallback simple features = daily returns
            self.features = self._default_features(self.prices)

        # Daily returns
        self.returns = self.prices.pct_change().fillna(0.0)

        self.dates = self.features.index
        self.n_steps = len(self.dates)
        self.n_assets = self.prices.shape[1]
        self.n_features = self.features.shape[1]

        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.drawdown_penalty = drawdown_penalty
        self.max_weight_per_asset = max_weight_per_asset

        # Observation = feature vector (one row of features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # Action = raw portfolio weights (projected later)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # Internal state
        self._current_step = 0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._weights = np.full(self.n_assets, 1.0 / self.n_assets)

    # -----------------------------
    # Helpers
    # -----------------------------

    def _default_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        # Simple fallback: each feature = daily return of each asset
        return prices.pct_change().fillna(0.0)

    def _get_observation(self) -> np.ndarray:
        return self.features.iloc[self._current_step].values.astype(np.float32)

    def _project_weights(self, action: np.ndarray) -> np.ndarray:
        # Enforce: w_i >= 0, sum(w_i)=1, w_i <= max_cap
        w = np.maximum(action.astype(np.float32), 0.0)

        # If all zeros : uniform weights
        if w.sum() <= 0:
            return np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)

        # Normalize
        w = w / w.sum()
        cap = float(self.max_weight_per_asset)

        # Iterative capping + redistribution
        while True:
            over = w > cap + 1e-8
            if not over.any():
                break

            excess = (w[over] - cap).sum()
            w[over] = cap

            under = ~over
            if not under.any():
                w = w / w.sum()
                break

            w[under] += excess / under.sum()

        w = np.maximum(w, 0.0)
        w = w / w.sum() if w.sum() > 0 else np.full(self.n_assets, 1.0 / self.n_assets)
        return w.astype(np.float32)

    # ------------------------
    # Gym API
    # ------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._weights = np.full(self.n_assets, 1.0 / self.n_assets)

        obs = self._get_observation()
        info = {
            "portfolio_value": self._portfolio_value,
            "step": 0,
            "date": self.dates[0],
            "drawdown": 0.0,
        }
        return obs, info

    def step(self, action: np.ndarray):
        # Project raw action to valid weights
        new_weights = self._project_weights(action)

        # Turnover = L1 distance between old & new weights
        turnover = float(np.abs(new_weights - self._weights).sum())
        cost = self.transaction_cost * turnover

        # Current step return
        date = self.dates[self._current_step]
        asset_returns = self.returns.loc[date].values.astype(np.float32)

        portfolio_return = float(np.dot(new_weights, asset_returns))
        net_return = portfolio_return - cost

        # Update portfolio value
        self._portfolio_value *= (1.0 + net_return)

        # Compute drawdown
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value
        drawdown = float(self._portfolio_value / self._peak_value - 1.0)

        # Reward = log-return + drawdown penalty
        net_return_clipped = max(net_return, -0.999)
        reward = float(np.log(1.0 + net_return_clipped)) + self.drawdown_penalty * drawdown

        # Advance state
        self._weights = new_weights
        self._current_step += 1

        terminated = self._current_step >= self.n_steps - 1
        truncated = False

        if terminated:
            obs = np.zeros(self.n_features, dtype=np.float32)
            next_date = self.dates[self._current_step - 1]
        else:
            obs = self._get_observation()
            next_date = self.dates[self._current_step]

        info = {
            "portfolio_value": self._portfolio_value,
            "portfolio_return": portfolio_return,
            "net_return": net_return,
            "turnover": turnover,
            "drawdown": drawdown,
            "step": self._current_step,
            "date": next_date,
            "weights": self._weights.copy(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        # readable debug print
        w = np.round(self._weights, 3)
        print(f"Step {self._current_step} | Value={self._portfolio_value:.4f} | Weights={w}")
