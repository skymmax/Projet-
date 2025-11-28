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
    - Reward: financial reward based on log-return net of transaction costs
              and a penalty on drawdown.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        initial_capital: float = 1.0,
        transaction_cost: float = 0.0,
        drawdown_penalty: float = 0.1,     # alpha: strength of drawdown penalty
        max_weight_per_asset: float = 0.4, # allocation constraint (e.g. 40% max)
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
        self.drawdown_penalty = drawdown_penalty
        self.max_weight_per_asset = max_weight_per_asset

        # Observation: feature vector for one date
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # Action: weights per asset (continuous [0, 1] before projection)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # Internal state
        self._current_step: int = 0
        self._portfolio_value: float = self.initial_capital
        self._peak_value: float = self.initial_capital  # for drawdown
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

    def _project_weights_with_cap(self, action: np.ndarray) -> np.ndarray:
        """
        Project raw action onto the feasible set:
            w_i >= 0, sum(w_i) = 1, w_i <= max_weight_per_asset
        using a simple iterative capping + redistribution scheme.
        """
        n = self.n_assets
        w = np.asarray(action, dtype=np.float32)

        # Step 1: ensure non-negative
        w = np.maximum(w, 0.0)

        # If everything is zero -> uniform
        if w.sum() <= 0:
            return np.full(n, 1.0 / n, dtype=np.float32)

        # Step 2: normalize to sum = 1
        w = w / w.sum()

        cap = float(self.max_weight_per_asset)

        # Step 3: iterative capping and redistribution
        # (n is small, so a simple loop is fine)
        while True:
            over_cap = w > cap + 1e-8
            if not over_cap.any():
                break

            # Excess mass above the cap
            excess = (w[over_cap] - cap).sum()

            # Cap the ones above the limit
            w[over_cap] = cap

            # Redistribute excess on remaining assets
            under_cap = ~over_cap
            if not under_cap.any():
                # All assets capped: renormalize and break
                w = w / w.sum()
                break

            w[under_cap] += excess / under_cap.sum()

        # Final clean-up: clip small negatives and renormalize
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.full(n, 1.0 / n, dtype=np.float32)
        else:
            w = w / w.sum()

        return w.astype(np.float32)

    # ---------- Gymnasium API ----------

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._weights = np.full(self.n_assets, 1.0 / self.n_assets)

        obs = self._get_observation()
        info = {
            "portfolio_value": self._portfolio_value,
            "step": self._current_step,
            "date": self.dates[self._current_step],
            "drawdown": 0.0,
        }
        return obs, info

    def step(self, action: np.ndarray):
        # Project action into feasible portfolio weights
        new_weights = self._project_weights_with_cap(action)

        # Transaction costs based on turnover (L1 distance between weight vectors)
        turnover = float(np.abs(new_weights - self._weights).sum())
        cost = self.transaction_cost * turnover

        # Current date and asset returns
        current_date = self.dates[self._current_step]
        asset_returns = self.returns.loc[current_date].values.astype(np.float32)

        # Portfolio return before costs
        portfolio_return = float(np.dot(new_weights, asset_returns))

        # Net return including transaction cost
        net_return = portfolio_return - cost

        # Update portfolio value
        self._portfolio_value *= (1.0 + net_return)

        # Update peak value and compute drawdown
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value

        drawdown = float(self._portfolio_value / self._peak_value - 1.0)

        # Compute reward: log-return + penalty on drawdown
        net_return_clipped = max(net_return, -0.999)  # avoid log(<=0)
        log_reward = float(np.log(1.0 + net_return_clipped))
        reward = log_reward + self.drawdown_penalty * drawdown

        # Update internal state
        self._weights = new_weights
        self._current_step += 1

        terminated = self._current_step >= (self.n_steps - 1)
        truncated = False

        if not terminated:
            obs = self._get_observation()
            info_date = self.dates[self._current_step]
        else:
            obs = np.zeros(self.n_features, dtype=np.float32)
            info_date = self.dates[self._current_step - 1]

        info = {
            "portfolio_value": self._portfolio_value,
            "portfolio_return": portfolio_return,
            "net_return": net_return,
            "turnover": turnover,
            "drawdown": drawdown,
            "step": self._current_step,
            "date": info_date,
            "weights": self._weights.copy(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Step {self._current_step} | "
            f"Value={self._portfolio_value:.4f} | "
            f"Weights={np.round(self._weights, 3)}"
        )
