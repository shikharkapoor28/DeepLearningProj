import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class TradingEnv(gym.Env):
    """
    Gymnasium environment for RL trading.

    Two modes:
    - **Synthetic** (default): no price series; observations are zeros; useful for API stubs / UI.
    - **Real data**: pass aligned `feature_matrix` [T, F] and `close_prices` [T] (e.g. from
      `DatasetBuilder` + Yahoo fetch + `FeatureEngineer`). Each `step` advances one bar, builds a
      `seq_len`×F window, and updates portfolio using realized returns and turnover costs.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_matrix: Optional[np.ndarray] = None,
        close_prices: Optional[np.ndarray] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        super().__init__()

        self.config = config or {}
        self.initial_balance = float(self.config.get("initial_balance", 100000.0))
        self.trading_fee = float(self.config.get("trading_fee_bps", 6)) / 10000.0
        self.slippage = float(self.config.get("slippage_bps", 0)) / 10000.0
        self.reward_scaling = float(self.config.get("reward_scaling", 1.0))
        self.max_drawdown_stop = float(self.config.get("max_drawdown_stop", 0.30))

        self.n_assets = int(self.config.get("n_assets", 1))
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.seq_len = int(self.config.get("seq_len", 64))

        self._feature_matrix: Optional[np.ndarray] = None
        self._close_prices: Optional[np.ndarray] = None
        self._feature_columns = feature_columns

        if feature_matrix is not None or close_prices is not None:
            if feature_matrix is None or close_prices is None:
                raise ValueError("Provide both feature_matrix and close_prices, or neither.")
            fm = np.asarray(feature_matrix, dtype=np.float32)
            cp = np.asarray(close_prices, dtype=np.float64).reshape(-1)
            if fm.shape[0] != cp.shape[0]:
                raise ValueError(
                    f"feature_matrix rows ({fm.shape[0]}) must match len(close_prices) ({cp.shape[0]})."
                )
            if fm.shape[0] < self.seq_len + 1:
                raise ValueError(
                    f"Need at least seq_len+1={self.seq_len + 1} rows for one market step; got {fm.shape[0]}."
                )
            if not np.all(cp > 0):
                raise ValueError("close_prices must be positive.")

            self._feature_matrix = fm
            self._close_prices = cp
            n_features = fm.shape[1]
        else:
            n_features = int(self.config.get("n_features", 10))

        self.config["n_features"] = n_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.seq_len, n_features),
            dtype=np.float32,
        )

        self.state: Optional[np.ndarray] = None
        self.portfolio_value = self.initial_balance
        self.weight = 0.0
        self._data_idx: int = 0
        self.episode_step: int = 0
        self.peak_value: float = self.initial_balance

    def _uses_real_data(self) -> bool:
        return self._feature_matrix is not None

    def _observation_at(self, end_idx: int) -> np.ndarray:
        """Window [end_idx - seq_len + 1, end_idx + 1) along time axis."""
        start = end_idx - self.seq_len + 1
        return self._feature_matrix[start : end_idx + 1].astype(np.float32, copy=False)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if "feature_matrix" in options or "close_prices" in options:
            fm = options.get("feature_matrix")
            cp = options.get("close_prices")
            if (fm is None) != (cp is None):
                raise ValueError("reset options: supply both feature_matrix and close_prices.")
            if fm is not None:
                self._feature_matrix = np.asarray(fm, dtype=np.float32)
                self._close_prices = np.asarray(cp, dtype=np.float64).reshape(-1)
                if self._feature_matrix.shape[0] != self._close_prices.shape[0]:
                    raise ValueError("feature_matrix and close_prices length mismatch.")
                if self._feature_matrix.shape[0] < self.seq_len + 1:
                    raise ValueError(
                        f"Need at least seq_len+1={self.seq_len + 1} rows for one market step."
                    )
                n_features = self._feature_matrix.shape[1]
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.seq_len, n_features),
                    dtype=np.float32,
                )

        self.portfolio_value = self.initial_balance
        self.weight = 0.0
        self.episode_step = 0
        self.peak_value = self.initial_balance

        if self._uses_real_data():
            self._data_idx = self.seq_len - 1
            self.state = self._observation_at(self._data_idx)
        else:
            self._data_idx = 0
            self.state = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.portfolio_value * (1.0 - self.weight),
            "holdings": self.portfolio_value * self.weight,
            "data_idx": self._data_idx,
        }
        return self.state, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        w_target = float(np.clip(action[0], 0.0, 1.0))

        terminated = False
        truncated = False
        prev_value = self.portfolio_value

        if self._uses_real_data():
            assert self._close_prices is not None
            i = self._data_idx
            if i >= len(self._close_prices) - 1:
                truncated = True
                reward = 0.0
                if self.state is None:
                    self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                ret = float(self._close_prices[i + 1] / self._close_prices[i] - 1.0)
                self.portfolio_value *= self.weight * (1.0 + ret) + (1.0 - self.weight)

                w_after_return = (
                    (self.weight * (1.0 + ret))
                    / (self.weight * (1.0 + ret) + (1.0 - self.weight) + 1e-12)
                )
                turnover = abs(w_target - w_after_return)
                cost_rate = turnover * (self.trading_fee + self.slippage)
                self.portfolio_value *= max(0.0, 1.0 - cost_rate)
                self.weight = w_target

                self._data_idx += 1
                self.episode_step += 1

                if self._data_idx >= len(self._close_prices) - 1:
                    truncated = True

                self.state = self._observation_at(self._data_idx)

                reward = (
                    np.log((self.portfolio_value + 1e-12) / (prev_value + 1e-12))
                    * self.reward_scaling
                )

                self.peak_value = max(self.peak_value, self.portfolio_value)
                dd = 1.0 - self.portfolio_value / (self.peak_value + 1e-12)
                if dd >= self.max_drawdown_stop:
                    terminated = True
        else:
            turnover = abs(w_target - self.weight)
            cost_rate = turnover * (self.trading_fee + self.slippage)
            self.portfolio_value *= max(0.0, 1.0 - cost_rate)
            self.weight = w_target

            self.episode_step += 1
            self.state = np.zeros(self.observation_space.shape, dtype=np.float32)

            reward = (
                np.log((self.portfolio_value + 1e-12) / (prev_value + 1e-12))
                * self.reward_scaling
            )

            if self.episode_step >= int(self.config.get("max_episode_steps", 100)):
                truncated = True

        info: Dict[str, Any] = {
            "step": self.episode_step,
            "executed_trade": {"target_weight": w_target},
            "portfolio_value": self.portfolio_value,
            "cash": self.portfolio_value * (1.0 - self.weight),
            "holdings": self.portfolio_value * self.weight,
            "data_idx": self._data_idx,
        }
        if self._feature_columns:
            info["feature_columns"] = self._feature_columns

        return self.state, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass
