"""
Central training engine for the RL trading system.

Uses Stable-Baselines3 PPO with a custom TransformerFeaturesExtractor
so the policy processes sequential market observations through multi-head
self-attention before the actor-critic MLP heads.

Training metrics are logged to a CSV file for later analysis by
research/validate_training.py.
"""

import os
import csv
import yaml
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from typing import Optional

from rl_core.ppo_agent import TransformerFeaturesExtractor


def get_transformer_policy_kwargs(
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
) -> dict:
    """
    Returns the policy_kwargs dict that tells SB3's MlpPolicy to use
    TransformerFeaturesExtractor as its feature backbone.
    """
    return dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ),
    )


class CSVLoggerCallback(BaseCallback):
    """
    Logs training metrics to a CSV file every `log_freq` timesteps.
    Produces the file that research/validate_training.py reads.
    """

    def __init__(self, log_path: str, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_freq = log_freq
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._file = open(self.log_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestep",
            "rollout/ep_rew_mean",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
        ])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            logger = self.model.logger.name_to_value
            self._writer.writerow([
                self.num_timesteps,
                logger.get("rollout/ep_rew_mean", ""),
                logger.get("train/policy_gradient_loss", ""),
                logger.get("train/value_loss", ""),
                logger.get("train/entropy_loss", ""),
            ])
            self._file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()


class RLTrainer:
    """
    Central training engine for Trading RL systems.

    Constructs PPO with TransformerFeaturesExtractor via policy_kwargs,
    ensuring the Transformer is part of the actual forward pass during
    training, evaluation, saving, and loading.
    """

    def __init__(self, env: gym.Env, config_path: str = "configs/default.yaml"):
        self.env = env

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        rl_settings = self.config.get("rl_algorithm", {})
        algo_name = rl_settings.get("name", "PPO").upper()
        hyperparams = rl_settings.get("hyperparameters", {})

        arch_settings = self.config.get("model_architecture", {})
        d_model = arch_settings.get("d_model", 128)
        n_heads = arch_settings.get("n_heads", 4)
        n_layers = arch_settings.get("n_layers", 2)

        policy_kwargs = get_transformer_policy_kwargs(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        if algo_name != "PPO":
            raise NotImplementedError(f"Algorithm {algo_name} not supported.")

        # Resolve log directory relative to repo root (best-effort)
        self._log_dir = str(
            Path(config_path).resolve().parent / "experiments" / "runs"
        )

        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            # PPO Hyperparameters (with typical RL defaults)
            learning_rate=hyperparams.get("learning_rate", 3e-4),
            n_steps=hyperparams.get("n_steps", 2048),        # Rollout length before policy update
            batch_size=hyperparams.get("batch_size", 256),   # Minibatch size for optimization
            n_epochs=hyperparams.get("n_epochs", 10),        # Number of epochs to train on the rollout
            gamma=hyperparams.get("gamma", 0.99),            # Discount factor for future rewards
            gae_lambda=hyperparams.get("gae_lambda", 0.95),  # Advantage estimation smoothing factor
            clip_range=hyperparams.get("clip_range", 0.2),   # PPO clipping mechanism to prevent large updates
            ent_coef=hyperparams.get("ent_coef", 0.01),      # Entropy coefficient to encourage exploration
            verbose=1,
            tensorboard_log=self._log_dir if self._has_tensorboard() else None,
        )

    @staticmethod
    def _has_tensorboard() -> bool:
        try:
            import tensorboard  # noqa: F401
            return True
        except Exception:
            return False

    def train(self, total_timesteps: int = 100000):
        """Execute a basic training loop with CSV logging."""
        csv_path = os.path.join(self._log_dir, "ppo_training_log.csv")
        csv_cb = CSVLoggerCallback(log_path=csv_path)
        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=csv_cb)
        print("Training complete.")

    def train_with_eval(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        eval_freq: int = 10_000,
        best_model_save_path: str = "experiments/checkpoints/best",
    ):
        """
        Train with periodic evaluation, best-model checkpointing, and CSV logging.
        """
        os.makedirs(best_model_save_path, exist_ok=True)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=best_model_save_path,
            eval_freq=max(1, int(eval_freq)),
            deterministic=True,
            render=False,
        )

        csv_path = os.path.join(self._log_dir, "ppo_training_log.csv")
        csv_cb = CSVLoggerCallback(log_path=csv_path)

        callback = CallbackList([csv_cb, eval_cb])
        print(f"Starting training for {total_timesteps} timesteps with eval_freq={eval_freq}...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        print("Training complete.")

    def save_model(self, path: str):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a saved model. custom_objects registers the Transformer extractor
        so SB3 can deserialize models that used it.
        """
        self.model = PPO.load(
            path,
            env=self.env,
            custom_objects={
                "policy_kwargs": get_transformer_policy_kwargs(),
            },
        )
        print(f"Model loaded from {path}")
