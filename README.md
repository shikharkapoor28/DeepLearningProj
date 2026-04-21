# Transformer-PPO Reinforcement Learning Trading System

A production-grade reinforcement learning trading system that uses a **Transformer-based feature extractor** inside a **Proximal Policy Optimization (PPO)** agent to learn portfolio allocation strategies on US equities (SPY).

## Architecture Summary

The system implements a custom `TransformerFeaturesExtractor` (a proper Stable-Baselines3 `BaseFeaturesExtractor` subclass) that processes sequential market observations through multi-head self-attention before feeding the pooled temporal representation to PPO's actor-critic MLP heads.

### Why Transformer + PPO?

- **Temporal dependencies**: Financial time series exhibit complex temporal patterns (momentum, mean-reversion, regime changes). The Transformer's self-attention mechanism learns which historical timesteps are most relevant to the current decision — unlike a standard MLP which sees only a flattened feature vector.
- **PPO stability**: PPO provides stable policy gradient updates via clipped surrogate objectives and is well-suited for continuous action spaces (portfolio weight allocation).
- **Explainability**: The attention weights from the Transformer's last layer are cached after each forward pass, providing interpretable insight into which time steps the agent considers most important.

### Model Architecture

```
Observation [batch, 64, 6]   ← 64-bar window × 6 features
        │
  Linear Projection (6 → 128)
        │
  + Positional Encoding (learnable)
        │
  TransformerEncoder (2 layers, 4 heads, d_model=128)
        │
  Mean Pooling → [batch, 128]
        │
  ┌─────┴─────┐
  Actor MLP    Critic MLP
  (→ action)   (→ value)
```

**Features**: return_1p, volatility(20), RSI(14), MACD(12,26,9), turbulence (vol z-score), Volume — all normalized via robust tanh normalization (median/MAD).

## Folder Structure

```
DeepLearningProj/
├── backend/
│   ├── main.py                    # FastAPI server + live inference loop
│   ├── train_policy.py            # Main training entry point
│   ├── rl_core/
│   │   ├── ppo_agent.py           # TransformerFeaturesExtractor (SB3 BaseFeaturesExtractor)
│   │   ├── trainer.py             # RLTrainer with CSV logging + Transformer policy_kwargs
│   │   └── explainability.py      # Real attention + gradient-based feature importance
│   ├── environment/
│   │   └── trading_env.py         # Gymnasium env with log-return reward + transaction costs
│   ├── evaluation/
│   │   ├── evaluator.py           # Backtest + financial metrics (252-day annualization)
│   │   └── baselines.py           # Buy-and-Hold + Random baselines
│   ├── data_pipeline/
│   │   ├── fetcher.py             # Yahoo Finance data downloader
│   │   └── feature_utils.py       # Feature engineering + normalization
│   ├── broker/                    # Paper broker for simulated execution
│   ├── research/
│   │   ├── optuna_tune.py         # Hyperparameter tuning (same Transformer architecture)
│   │   ├── validate_training.py   # Training curve plots from real CSV logs
│   │   ├── compare_baselines.py   # RL vs Buy&Hold vs Random comparison
│   │   ├── reward_sanity.py       # Reward structure sanity checks
│   │   └── run_ablations.py       # Ablation study framework
│   └── requirements.txt
├── configs/
│   └── default.yaml               # Hyperparameters, architecture config
├── experiments/
│   ├── checkpoints/               # Saved model weights (.zip)
│   ├── runs/                      # Training logs (CSV, TensorBoard)
│   └── reports/                   # Generated plots (PNG)
├── frontend/                      # Next.js real-time dashboard
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend dashboard)

### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Optuna is required for hyperparameter tuning (installed separately):
```bash
pip install optuna
```

## Usage

### 1. Train the Model

Trains PPO with the Transformer extractor on historical SPY data (2001–2026), with an 80/20 train/validation split and periodic evaluation checkpointing:

```bash
cd backend
python train_policy.py --ticker SPY --timesteps 1000000
```

**Key arguments:**
| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `SPY` | Yahoo Finance ticker symbol |
| `--start` | `2001-01-01` | Historical data start date |
| `--end` | `2026-04-19` | Historical data end date |
| `--timesteps` | `1000000` | Total PPO training timesteps |
| `--val-ratio` | `0.2` | Fraction of data for validation |
| `--eval-freq` | `10000` | Timesteps between evaluations |

**Outputs:**
- Final model: `experiments/checkpoints/ppo_spy.zip`
- Best model: `experiments/checkpoints/best/best_model.zip`
- Training log: `experiments/runs/ppo_training_log.csv`

### 2. Hyperparameter Tuning (Optuna)

Runs 20 trials of 50k timesteps each, using the same Transformer architecture:

```bash
cd backend
python research/optuna_tune.py
```

After tuning, update `configs/default.yaml` with the best hyperparameters and retrain.

### 3. Generate Training Validation Plots

Reads the real training CSV log and generates diagnostic plots:

```bash
cd backend
python research/validate_training.py
```

Output: `experiments/reports/training_validation.png`

> **Note:** This script requires training to have been run first. It will not generate synthetic/dummy data.

### 4. Run Baseline Comparison

Compares the trained RL agent against Buy-and-Hold and Random baselines on the validation set:

```bash
cd backend
python research/compare_baselines.py
```

Output: `experiments/reports/baseline_comparison.png`

### 5. Run Live Dashboard

Start the backend (FastAPI + WebSocket):
```bash
cd backend
python main.py
```

Start the frontend (Next.js):
```bash
cd frontend
npm install
npm run dev
```

To use a trained model for live inference:
```bash
# On Windows
set PPO_MODEL_PATH=experiments/checkpoints/ppo_spy.zip
cd backend
python main.py

# On Unix
export PPO_MODEL_PATH="experiments/checkpoints/ppo_spy.zip"
cd backend && python main.py
```

## Configuration

All hyperparameters and architecture settings are in `configs/default.yaml`:

```yaml
rl_algorithm:
  name: "PPO"
  hyperparameters:
    learning_rate: 5.476e-05   # Tuned via Optuna
    n_steps: 512
    batch_size: 64
    gamma: 0.9847
    ent_coef: 0.0336

model_architecture:
  encoder: "transformer"
  seq_len: 64
  d_model: 128
  n_heads: 4
  n_layers: 2
```

## Evaluation Metrics

All financial metrics use **252 trading days** for annualization (US equities convention):

| Metric | Formula |
|--------|---------|
| Cumulative Return | (final − initial) / initial |
| Sharpe Ratio | (mean_return / std_return) × √252 |
| Sortino Ratio | (mean_return / downside_std) × √252 |
| Max Drawdown | max((peak − value) / peak) |

## Explainability

The system provides two real explainability mechanisms:

1. **Attention Weights**: Extracted from the last Transformer encoder layer after each forward pass. Shows which historical timesteps the model considers most relevant.
2. **Gradient-based Feature Importance**: Computes input sensitivity via backpropagation through the critic's value head. Indicates which features (Return, Volatility, RSI, MACD, Turbulence, Volume) most influence the value estimate.

Both are streamed to the frontend dashboard during live inference when a trained model is loaded.

## Known Limitations

- **Single-asset only**: The current environment supports one asset (SPY). Multi-asset portfolio optimization would require extending the action space and environment.
- **Walk-forward validation**: Not implemented. The current evaluation uses a simple train/val temporal split. A rolling-window walk-forward validation would better assess generalization.
- **Ablation studies**: The `run_ablations.py` script produces illustrative comparisons. A production version would automate training runs across architecture variants.
- **Live inference frequency**: The dashboard streams at ~1.5s intervals using Yahoo Finance API. This is for demonstration; production would require a proper market data feed.
