"""
Optuna hyperparameter optimization for PPO with TransformerFeaturesExtractor.

Tunes the same architecture used in the main training pipeline: PPO with
TransformerFeaturesExtractor, not a bare MlpPolicy. This ensures that
tuned hyperparameters are valid for the real model.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
import numpy as np
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from data_pipeline.fetcher import YahooFinanceFetcher
from data_pipeline.feature_utils import (
    build_feature_matrix,
    compute_features,
    flatten_yfinance_columns,
)
from environment.trading_env import TradingEnv
from rl_core.ppo_agent import TransformerFeaturesExtractor
from rl_core.trainer import get_transformer_policy_kwargs

warnings.filterwarnings("ignore")


def optimize_ppo(trial):
    """
    Optuna objective: create a PPO with Transformer extractor, train for
    a short horizon, and evaluate on the validation set.
    """
    # 1. Hyperparameter sampling
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)

    # PPO constraint: n_steps must be divisible by batch_size
    if n_steps % batch_size != 0:
        n_steps = (n_steps // batch_size) * batch_size
        if n_steps == 0:
            n_steps = batch_size

    # 2. Build environments for this trial
    train_env = TradingEnv(
        config=env_config, feature_matrix=train_fm, close_prices=train_cp
    )
    val_env = TradingEnv(
        config=env_config, feature_matrix=val_fm, close_prices=val_cp
    )

    # 3. Model with Transformer extractor (same architecture as main training)
    policy_kwargs = get_transformer_policy_kwargs()

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        verbose=0,
    )

    # 4. Short training cycle to gauge parameter viability
    try:
        model.learn(total_timesteps=50_000)
    except Exception as e:
        print(f"Trial failed: {trial.params}. Error: {e}")
        return -np.inf

    # 5. Evaluate on validation set
    try:
        mean_reward, _ = evaluate_policy(
            model, val_env, n_eval_episodes=2, deterministic=True
        )
    except Exception:
        mean_reward = -np.inf

    return mean_reward


if __name__ == "__main__":
    print("Fetching SPY data for hyperparameter optimization...")
    fetcher = YahooFinanceFetcher(["SPY"])
    raw = fetcher.fetch_historical("2001-01-01", "2026-04-19", interval="1d")
    raw = flatten_yfinance_columns(raw)
    raw = raw[raw["Ticker"] == "SPY"]

    ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    feat_df = compute_features(ohlcv)
    feature_matrix, close_prices, _ = build_feature_matrix(feat_df)

    # Train/Val split (80/20)
    split = int(len(close_prices) * 0.8)

    # Module-level variables accessed by the objective function
    train_fm, val_fm = feature_matrix[:split], feature_matrix[split:]
    train_cp, val_cp = close_prices[:split], close_prices[split:]

    env_config = {
        "seq_len": 64,
        "initial_balance": 100000.0,
        "trading_fee_bps": 6,
        "slippage_bps": 2,
        "reward_scaling": 100.0,
        "max_drawdown_stop": 0.30,
        "n_assets": 1,
    }

    print("\nStarting Optuna study (20 trials × 50k timesteps each)...")
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study.optimize(optimize_ppo, n_trials=20)

    print("\n" + "=" * 60)
    print("Optimization finished!")
    print(f"Best trial reward: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 60)
    print(
        "\nTo use these values, update configs/default.yaml and re-run:"
    )
    print("  cd backend && python train_policy.py --timesteps 1000000")
