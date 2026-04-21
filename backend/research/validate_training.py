"""
Training curve validation.

Reads the real CSV training log produced by CSVLoggerCallback in
rl_core/trainer.py and generates diagnostic plots.

If no training log exists, this script prints an error with instructions
on how to generate one — it does NOT fabricate dummy data.
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Add backend/ to path so this can be run standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_training_log() -> str:
    """
    Searches for the training CSV log in standard locations relative
    to the repo root.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "experiments" / "runs" / "ppo_training_log.csv",
        repo_root / "backend" / "experiments" / "runs" / "ppo_training_log.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def validate_training_curves(log_csv_path: str = "") -> None:
    """
    Reads the CSV log and produces a 2×2 diagnostic figure:
      - Reward vs Timesteps
      - Policy Gradient Loss
      - Value Loss
      - Entropy Loss

    Saves the figure to experiments/reports/training_validation.png.
    """
    if not log_csv_path:
        log_csv_path = find_training_log()

    if not log_csv_path or not os.path.exists(log_csv_path):
        print("=" * 60)
        print("ERROR: No training log found.")
        print()
        print("This script requires real training data — it does not")
        print("generate synthetic curves.")
        print()
        print("To generate the log, run training first:")
        print("  cd backend && python train_policy.py --timesteps 100000")
        print()
        print("The trainer will produce:")
        print("  experiments/runs/ppo_training_log.csv")
        print()
        print("Then re-run this script:")
        print("  python research/validate_training.py")
        print("=" * 60)
        sys.exit(1)

    df = pd.read_csv(log_csv_path)
    print(f"Loaded training log: {log_csv_path} ({len(df)} rows)")

    # Drop rows where metrics are empty (logged before first update)
    numeric_cols = [c for c in df.columns if c != "timestep"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["rollout/ep_rew_mean"], inplace=True)

    if df.empty:
        print("WARNING: Training log exists but contains no valid metric rows.")
        print("Training may not have run long enough to produce rollout stats.")
        sys.exit(1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Validation — Real Metrics", fontsize=14, fontweight="bold")

    # Reward
    axs[0, 0].plot(df["timestep"], df["rollout/ep_rew_mean"], color="blue", linewidth=1)
    axs[0, 0].set_title("Mean Episode Reward")
    axs[0, 0].set_xlabel("Timesteps")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True, alpha=0.3)

    # Policy gradient loss
    if "train/policy_gradient_loss" in df.columns:
        valid = df.dropna(subset=["train/policy_gradient_loss"])
        axs[0, 1].plot(valid["timestep"], valid["train/policy_gradient_loss"], color="red", linewidth=1)
    axs[0, 1].set_title("Policy Gradient Loss")
    axs[0, 1].set_xlabel("Timesteps")
    axs[0, 1].grid(True, alpha=0.3)

    # Value loss
    if "train/value_loss" in df.columns:
        valid = df.dropna(subset=["train/value_loss"])
        axs[1, 0].plot(valid["timestep"], valid["train/value_loss"], color="green", linewidth=1)
    axs[1, 0].set_title("Value Loss")
    axs[1, 0].set_xlabel("Timesteps")
    axs[1, 0].grid(True, alpha=0.3)

    # Entropy loss
    if "train/entropy_loss" in df.columns:
        valid = df.dropna(subset=["train/entropy_loss"])
        axs[1, 1].plot(valid["timestep"], valid["train/entropy_loss"], color="purple", linewidth=1)
    axs[1, 1].set_title("Entropy Loss")
    axs[1, 1].set_xlabel("Timesteps")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "experiments" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_validation.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved training validation plot to {out_path}")


if __name__ == "__main__":
    # Accept optional explicit path as CLI argument
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    validate_training_curves(path)
