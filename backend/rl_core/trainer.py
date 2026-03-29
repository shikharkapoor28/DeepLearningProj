import os
import yaml
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.callbacks import BaseCallback

class TradingCallback(BaseCallback):
    """
    Custom callback for logging, saving checkpoints, and tracking metrics.
    """
    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        # Save model every N steps or track custom portfolio returns here
        if self.num_timesteps % 5000 == 0:
            # Example checkpointing
            # self.model.save(f"experiments/checkpoints/ppo_{self.num_timesteps}_steps")
            pass
        return True

class RLTrainer:
    """
    Central training engine for Trading RL systems.
    Handles PPO execution and integrates environment with models.
    """
    def __init__(self, env: gym.Env, config_path: str = "configs/default.yaml"):
        self.env = env
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        rl_settings = self.config.get("rl_algorithm", {})
        algo_name = rl_settings.get("name", "PPO").upper()
        hyperparams = rl_settings.get("hyperparameters", {})
        
        # Policy setup. For a custom network (like the Transformer one we scaffolded), 
        # we would pass policy_kwargs here. For now, using standard MlpPolicy as placeholder for Stable-Baselines3 integration.
        policy_kwargs = dict(
            features_extractor_class=None, # To be linked to TransformerEncoder
            features_extractor_kwargs=None,
        )
        
        if algo_name == "PPO":
            self.model = PPO(
                "MlpPolicy", 
                env, 
                learning_rate=hyperparams.get("learning_rate", 3e-4),
                n_steps=hyperparams.get("n_steps", 2048),
                batch_size=hyperparams.get("batch_size", 256),
                n_epochs=hyperparams.get("n_epochs", 10),
                gamma=hyperparams.get("gamma", 0.99),
                gae_lambda=hyperparams.get("gae_lambda", 0.95),
                clip_range=hyperparams.get("clip_range", 0.2),
                ent_coef=hyperparams.get("ent_coef", 0.01),
                verbose=1,
                tensorboard_log="experiments/runs/"
            )
        else:
            raise NotImplementedError(f"Algorithm {algo_name} is not fully integrated yet.")

    def train(self, total_timesteps: int = 100000):
        """Executes the training loop."""
        callback = TradingCallback()
        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        print("Training complete.")
        
    def save_model(self, path: str):
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
