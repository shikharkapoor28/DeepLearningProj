import numpy as np

class BuyAndHoldBaseline:
    """
    Simplest baseline: Allocates 100% of portfolio to the asset and holds.
    Used to compare if the RL agent actually learns a better strategy than doing nothing.
    """
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def predict(self, obs, deterministic=True):
        # Assuming action space is target weights: [Asset_1, ..., Asset_n]
        # We allocate fully to Asset 1
        action = np.zeros(self.action_dim)
        if self.action_dim > 0:
            action[0] = 1.0 # 100% weight to first asset
        return action, None


class RandomAgentBaseline:
    """
    Executes random actions. Used to test environment mechanics and bottom-tier baseline.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=False):
        return self.action_space.sample(), None

# Note: SAC and DQN baselines can be initialized directly using stable-baselines3 algorithms
# e.g., SAC('MlpPolicy', env) 
