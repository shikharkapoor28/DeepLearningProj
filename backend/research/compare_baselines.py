"""
Baseline comparison script.

Runs RL agent (loads trained model), Buy-and-Hold, and Random agent on
the validation split of real historical data, then plots equity curves
and key metrics.

Uses 252 trading days for annualization (US equities).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from evaluation.baselines import BuyAndHoldBaseline, RandomAgentBaseline
from evaluation.evaluator import Evaluator
from environment.trading_env import TradingEnv
from data_pipeline.fetcher import YahooFinanceFetcher
from data_pipeline.feature_utils import (
    build_feature_matrix,
    compute_features,
    flatten_yfinance_columns,
)
from rl_core.trainer import get_transformer_policy_kwargs


def run_comparison():
    repo_root = Path(__file__).resolve().parents[2]

    # Load real historical data
    print("Fetching historical SPY data...")
    fetcher = YahooFinanceFetcher(["SPY"])
    raw = fetcher.fetch_historical("2001-01-01", "2026-04-19", interval="1d")
    raw = flatten_yfinance_columns(raw)
    raw = raw[raw["Ticker"] == "SPY"]

    ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    feat_df = compute_features(ohlcv)
    feature_matrix, close_prices, _ = build_feature_matrix(feat_df)

    # Use the last 20% as validation (same split as training)
    split = int(len(close_prices) * 0.8)
    val_fm = feature_matrix[split:]
    val_cp = close_prices[split:]

    env_config = {
        "seq_len": 64,
        "initial_balance": 100000.0,
        "trading_fee_bps": 6,
        "slippage_bps": 2,
        "reward_scaling": 100.0,
        "max_drawdown_stop": 0.30,
        "n_assets": 1,
    }

    # Try to load a trained model; fall back to rule-based if none exists
    model_path = repo_root / "experiments" / "checkpoints" / "best" / "best_model.zip"
    if not model_path.exists():
        model_path = repo_root / "experiments" / "checkpoints" / "ppo_spy.zip"

    rl_agent = None
    if model_path.exists():
        print(f"Loading trained model from {model_path}...")
        rl_agent = PPO.load(
            str(model_path),
            custom_objects={"policy_kwargs": get_transformer_policy_kwargs()},
        )
    else:
        print("WARNING: No trained model found. Run train_policy.py first.")
        print(f"Looked for: {model_path}")
        print("Proceeding with baselines only.\n")

    agents = {}
    if rl_agent is not None:
        agents["PPO + Transformer"] = rl_agent
    agents["Buy & Hold"] = BuyAndHoldBaseline(action_dim=1)
    agents["Random"] = RandomAgentBaseline(
        action_space=TradingEnv(config=env_config, feature_matrix=val_fm, close_prices=val_cp).action_space
    )

    results = {}
    curves = {}

    for name, agent in agents.items():
        env = TradingEnv(config=env_config, feature_matrix=val_fm, close_prices=val_cp)
        evaluator = Evaluator(env)
        metrics = evaluator.run_backtest(agent)

        # Also capture equity curve for plotting
        env2 = TradingEnv(config=env_config, feature_matrix=val_fm, close_prices=val_cp)
        obs, _ = env2.reset()
        pvs = [env2.portfolio_value]
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env2.step(action)
            done = term or trunc
            pvs.append(env2.portfolio_value)
        curves[name] = pvs
        results[name] = metrics

    df_results = pd.DataFrame(results).T
    print("\n--- Performance Table ---")
    print(df_results.to_string())

    # Plotting
    n_agents = len(agents)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Baseline Comparison — Real Validation Data", fontsize=13, fontweight="bold")

    # Equity Curves
    for name, curve in curves.items():
        axs[0].plot(curve, label=name)
    axs[0].set_title("Equity Curves")
    axs[0].set_ylabel("Portfolio Value ($)")
    axs[0].set_xlabel("Steps")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    colors = ["#2196F3", "#9E9E9E", "#F44336"][:n_agents]

    if "sharpe_ratio" in df_results.columns:
        df_results["sharpe_ratio"].plot(kind="bar", ax=axs[1], color=colors)
        axs[1].set_title("Sharpe Ratio (annualized, 252d)")
        axs[1].grid(True, alpha=0.3)

    if "max_drawdown" in df_results.columns:
        df_results["max_drawdown"].plot(kind="bar", ax=axs[2], color=colors)
        axs[2].set_title("Max Drawdown")
        axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = repo_root / "experiments" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_comparison.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"\nSaved baseline comparison to {out_path}")


if __name__ == "__main__":
    run_comparison()
