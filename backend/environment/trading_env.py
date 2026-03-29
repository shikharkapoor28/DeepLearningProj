import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    """
    OpenAI Gymnasium environment for RL Trading.
    Simulates portfolio management and market frictions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super(TradingEnv, self).__init__()
        
        # Configuration
        self.config = config or {}
        self.initial_balance = self.config.get("initial_balance", 100000.0)
        self.trading_fee = self.config.get("trading_fee_bps", 6) / 10000.0
        
        # Asset configuration (e.g. BTC, ETH vs Cash)
        self.n_assets = self.config.get("n_assets", 1)  # Default 1 asset + Cash
        
        # Action space: target weight per asset (softmaxed or continuous) between 0 and 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        
        # Observation space: 
        # Seq_len (e.g., 64) x Num_features (e.g., 10 indicators)
        seq_len = self.config.get("seq_len", 64)
        n_features = self.config.get("n_features", 10)
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(seq_len, n_features), 
            dtype=np.float32
        )
        
        # Internal State
        self.state = None
        self.portfolio_value = self.initial_balance
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.portfolio_value = self.initial_balance
        self.current_step = 0
        
        # TODO: Initialize internal state from data pipeline
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.initial_balance,
            "holdings": 0.0
        }
        return self.state, info

    def step(self, action):
        # 1. Simulate trade execution based on action
        # 2. Calculate friction (slippage/fees)
        # 3. Update portfolio value
        
        reward = 0.0 
        terminated = False
        truncated = False
        
        # TODO: Fetch next state from data pipeline
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        self.current_step += 1
        
        if self.current_step >= 100:
            truncated = True
        
        info = {
            "step": self.current_step,
            "executed_trade": {}, # Trade details for explanation/streaming
            "portfolio_value": self.portfolio_value
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self):
        """Used for visual rendering if needed."""
        pass
