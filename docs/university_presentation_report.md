# Deep Reinforcement Learning for Algorithmic Trading: Project Report

## 1. Introduction
This project explores the application of Deep Reinforcement Learning (DRL) to algorithmic trading. The primary objective is to train an autonomous trading agent capable of navigating financial markets, managing a portfolio, and maximizing returns while adhering to strict risk management constraints.

We utilized Proximal Policy Optimization (PPO), a state-of-the-art policy gradient method, coupled with a custom Transformer-based feature extractor to process sequential market data.

---

## 2. Dataset and Feature Engineering

### 2.1 Dataset Selection
To ensure the robustness of our model, we required a large, historically significant dataset. We utilized daily historical trading data for the **S&P 500 ETF Trust (SPY)**. 

* **Timeframe:** 25 Years (January 2001 to April 2026)
* **Data Points:** ~6,300 daily rows (Open, High, Low, Close, Volume)
* **Source:** Yahoo Finance API (`yfinance`) and static local caches (`spy_historical_dataset_2010.csv`).

> **Note:** 25 years of daily market data encompasses multiple macroeconomic cycles, including the 2008 Financial Crisis, the 2020 COVID-19 crash, and subsequent bull markets. This ensures the agent is trained on a highly diverse set of market conditions, preventing overfitting to recent trends.

### 2.2 Feature Engineering Pipeline
Raw price and volume data alone are insufficient for robust market prediction. Our data pipeline systematically engineers a comprehensive state matrix for the RL agent. For every single timestep, the following technical indicators are computed and normalized:

1. **Trend Indicators:** Moving Average Convergence Divergence (MACD), Simple Moving Averages.
2. **Momentum Indicators:** Relative Strength Index (RSI - 14 period).
3. **Volatility Indicators:** Average True Range (ATR), Bollinger Bands.
4. **Returns:** Log returns and normalized close returns.

These features are computed across the entire dataset, creating a dense, multi-dimensional feature matrix that serves as the "environment state" observed by the agent.

---

## 3. Model Architecture

The core of our trading agent is built using the **Stable Baselines 3** implementation of **Proximal Policy Optimization (PPO)**. 

### 3.1 Custom Transformer Feature Extractor
Instead of relying on a standard Multi-Layer Perceptron (MLP), we implemented a custom `TransformerFeaturesExtractor`. Financial data is inherently sequential. The Transformer architecture allows the model to utilize self-attention mechanisms to weigh the importance of different historical timesteps within the agent's observation window (`seq_len = 64` days), effectively capturing long-term dependencies in market behavior that standard MLPs or LSTMs might miss.

### 3.2 The Trading Environment
We constructed a custom OpenAI `gymnasium` environment (`TradingEnv`) that simulates a real brokerage:
* **Initial Balance:** $100,000
* **Trading Fees:** 0.06% (Taker fee)
* **Slippage:** 0.02%
* **Risk Management:** The episode terminates early if a maximum drawdown of 30% is breached, teaching the agent to prioritize capital preservation.

---

## 4. Hyperparameter Tuning Methodology

Deep Learning models are highly sensitive to their hyperparameters. To find the optimal configuration for our PPO agent, we utilized **Optuna**, an advanced automatic hyperparameter optimization framework.

### 4.1 Optuna Search Space
We defined an objective function that trains the PPO model on an 80% training split and evaluates it on a 20% validation split. Optuna ran 20 distinct trials, searching across the following hyperparameter space:

| Hyperparameter | Search Space | Description |
| :--- | :--- | :--- |
| `learning_rate` | 1e-5 to 1e-3 (Log) | Step size for gradient descent updates. |
| `batch_size` | [64, 128, 256] | Number of samples processed before updating the model. |
| `n_steps` | [512, 1024, 2048] | Number of steps to run for each environment per update. |
| `gamma` | 0.90 to 0.999 | Discount factor; determines how much the agent cares about long-term vs. short-term rewards. |
| `ent_coef` | 0.0 to 0.05 | Entropy coefficient; encourages the agent to explore different actions to prevent premature convergence. |

### 4.2 Evaluation Metric
For each trial, the model was trained for 50,000 timesteps. The performance of the resulting policy was then evaluated deterministically across 2 episodes in the unseen validation environment. The Optuna objective was to **maximize the mean reward** returned during this validation phase.

---

## 5. Conclusion and Next Steps

By combining a massive 25-year financial dataset with a sophisticated Transformer-PPO architecture and rigorous Optuna hyperparameter tuning, we have developed a highly robust pipeline for training autonomous trading agents. 

The hyperparameter optimization process ensures that our learning rate, exploration rates, and batch sizes are statistically optimized for our specific dataset rather than relying on arbitrary defaults. The optimal hyperparameters discovered by Optuna are subsequently loaded into `configs/default.yaml` for the final, extended training run (1,000,000+ timesteps) to produce the production-ready model.
