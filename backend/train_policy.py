import argparse
import os
from pathlib import Path

import numpy as np

from data_pipeline.fetcher import YahooFinanceFetcher
from data_pipeline.feature_utils import (
    build_feature_matrix,
    compute_features,
    flatten_yfinance_columns,
)
from environment.trading_env import TradingEnv
from rl_core.trainer import RLTrainer
from evaluation.evaluator import Evaluator
from evaluation.baselines import BuyAndHoldBaseline, RandomAgentBaseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO policy on real Yahoo historical data.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-04-18")
    parser.add_argument("--interval", default="1d", help="yfinance interval: 1m,5m,15m,1h,1d...")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--out", default="experiments/checkpoints/ppo_spy")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="fraction of data reserved for validation")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="timesteps between eval runs")
    parser.add_argument("--best-dir", default="experiments/checkpoints/best", help="directory to save best model")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fetcher = YahooFinanceFetcher([args.ticker])
    raw = fetcher.fetch_historical(args.start, args.end, interval=args.interval)
    raw = flatten_yfinance_columns(raw)

    if raw.empty:
        raise SystemExit("No historical data returned. Check dates/interval/ticker.")

    # historical comes with a Ticker column + index as datetime
    if "Ticker" in raw.columns:
        raw = raw[raw["Ticker"] == args.ticker]

    ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    feat_df = compute_features(ohlcv)
    feature_matrix, close_prices, _ = build_feature_matrix(feat_df)

    if not (0.0 < args.val_ratio < 0.9):
        raise SystemExit("--val-ratio must be between 0 and 0.9")
    # Ensure both splits can run at least one env step.
    min_len = 64 + 1  # seq_len + 1
    total_len = len(close_prices)
    desired_val_len = int(round(total_len * args.val_ratio))
    val_len = max(min_len, min(total_len - min_len, desired_val_len))
    train_len = total_len - val_len
    if train_len < min_len or val_len < min_len:
        raise SystemExit(
            f"Not enough rows for train/val split with seq_len=64. "
            f"Need train_len>= {min_len} and val_len>= {min_len}; got train_len={train_len}, val_len={val_len}."
        )
    split = train_len

    train_fm, val_fm = feature_matrix[:split], feature_matrix[split:]
    train_cp, val_cp = close_prices[:split], close_prices[split:]

    config = {
        "seq_len": 64,
        "initial_balance": 100000.0,
        "trading_fee_bps": 6,
        "slippage_bps": 2,
        "reward_scaling": 100.0,
        "max_drawdown_stop": 0.30,
        "n_assets": 1,
    }

    train_env = TradingEnv(config=config, feature_matrix=train_fm, close_prices=train_cp)
    val_env = TradingEnv(config=config, feature_matrix=val_fm, close_prices=val_cp)

    # RLTrainer expects path relative to backend/; use repo_root/configs/default.yaml
    config_path = str((repo_root / "configs" / "default.yaml").resolve())
    trainer = RLTrainer(env=train_env, config_path=config_path)

    best_dir = (repo_root / args.best_dir).resolve()
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.train_with_eval(
        total_timesteps=args.timesteps,
        eval_env=val_env,
        eval_freq=args.eval_freq,
        best_model_save_path=str(best_dir),
    )

    trainer.save_model(str(out_path))

    # Evaluate: best checkpoint if present, else final model
    best_zip = best_dir / "best_model.zip"
    if best_zip.exists():
        trainer.load_model(str(best_zip))

    evaluator = Evaluator(val_env)
    rl_metrics = evaluator.run_backtest(trainer.model)

    bh = BuyAndHoldBaseline(action_dim=val_env.action_space.shape[0])
    rnd = RandomAgentBaseline(action_space=val_env.action_space)
    bh_metrics = evaluator.run_backtest(bh)
    rnd_metrics = evaluator.run_backtest(rnd)

    print()
    print("Validation metrics (deterministic episode):")
    print(f"  RL (PPO):     {rl_metrics}")
    print(f"  Buy&Hold:     {bh_metrics}")
    print(f"  Random:       {rnd_metrics}")

    print()
    print("To use this policy in live streaming:")
    print(f'  export PPO_MODEL_PATH="{out_path}.zip"')
    print("  cd backend && python main.py")


if __name__ == "__main__":
    main()

