# Full-Stack Deep Learning Trading System: Project Context Report

This document serves as a comprehensive summary of the "DeepLearningProj" codebase. It is designed to provide full context to any Large Language Model (LLM) so that it can generate a detailed university-level presentation, technical report, or documentation suite.

## 1. Project Overview
The project is a **Full-Stack Reinforcement Learning (RL) Trading System**. It trains a Proximal Policy Optimization (PPO) agent to trade assets (e.g., SPY) using real historical data and simulates live trading via a web dashboard. The system is split into a Python/FastAPI backend for RL inference and data processing, and a Next.js frontend for real-time visualization.

## 2. Technology Stack
*   **Backend**: Python, FastAPI, WebSockets, Uvicorn, Pandas, NumPy.
*   **Machine Learning / RL**: Stable-Baselines3 (PPO algorithm), PyTorch (Custom Transformer-based policy networks), OpenAI Gym (Custom Trading Environment).
*   **Data Pipeline**: `yfinance` API for historical and live market data.
*   **Frontend**: Next.js, React, Tailwind CSS (assumed based on Next.js ecosystem), WebSockets for real-time streaming.

## 3. Directory Structure & Key Components

### `backend/` (Core Logic & ML)
*   **`main.py`**: The primary FastAPI entrypoint. Orchestrates the live simulation, manages WebSocket connections, processes real-time data from Yahoo Finance, queries the PPO model for actions, and streams the portfolio state/actions to the frontend.
*   **`train_policy.py`**: A CLI script to fetch historical data, compute technical features, split data into train/val sets, train the PPO agent, and evaluate it against Buy-and-Hold and Random baselines.
*   **`data_pipeline/`**: Contains scripts to fetch data (`fetcher.py`) and compute technical indicators (RSI, MACD, Volatility, Turbulence) to build the feature matrix (`feature_utils.py`).
*   **`environment/`**: Contains `trading_env.py`, a custom Gym environment that simulates market friction (fees, slippage), tracks portfolio value, and calculates rewards based on portfolio returns and drawdown penalties.
*   **`rl_core/`**: Contains the RL training logic (`trainer.py`), custom transformer policies, and an `explainability.py` module to extract attention weights and feature importance from the RL model's decisions.
*   **`broker/`**: Contains `paper.py` for simulated execution of trades based on the RL agent's target portfolio weights.
*   **`evaluation/`**: Contains `evaluator.py` and `baselines.py` for backtesting and benchmarking the RL agent.

### `frontend/` (User Interface)
*   A Next.js application that connects to the backend via WebSockets (`ws://localhost:8000/ws/simulation/{session_id}`).
*   Displays real-time asset prices, portfolio value, agent actions (Buy/Sell/Hold), and explainability metrics (which features the AI is focusing on).

### `experiments/` (Models & Results)
*   **`checkpoints/`**: Stores the trained `.zip` model weights (e.g., `ppo_spy.zip`) and best-performing checkpoints.
*   **`reports/`**: Contains visual artifacts generated during training, such as ablation comparisons, baseline comparisons, and training validation plots (`.png` and `.csv`).

### `docs/` (Documentation)
*   Contains existing markdown architecture diagrams, API contracts, UI screen layouts, and a draft of the `university_presentation_report.md`.

## 4. System Data Flow
1.  **Training Phase**:
    *   `yfinance` -> `data_pipeline` (Features) -> `TradingEnv` -> `RLTrainer (PPO)` -> `experiments/checkpoints/best_model.zip`.
2.  **Live Simulation Phase**:
    *   User starts simulation via Frontend UI.
    *   `main.py` starts an async loop fetching 1-minute interval data from `yfinance`.
    *   Data flows into the feature pipeline to compute the current state.
    *   State is passed to the loaded PPO model to predict target portfolio weights.
    *   The `PaperBroker` simulates the trade execution (calculating fees/slippage).
    *   The `ExplainabilityLayer` extracts the model's attention.
    *   State, actions, portfolio metrics, and explanations are packaged and broadcasted over WebSockets to the Next.js frontend.

## 5. Recent Codebase Updates
*   **Optimization**: Removed all massive, transient dependencies and cache files (`node_modules/`, `.next/`, `__pycache__/`, `venv/`, `.venv/`) to maintain a clean directory structure.
*   **Code Polish**: Added comprehensive, human-like module and function-level docstrings to `backend/main.py` and `backend/train_policy.py` to ensure high readability and academic standard compliance.

## 6. Prompting Instructions for Next LLM
When generating the detailed report/presentation, please ensure you emphasize:
*   The integration between the custom Reinforcement Learning environment and the real-time web architecture.
*   The usage of Transformers in the policy network and the "Explainability" layer, which is a major academic talking point.
*   The realistic simulation of market mechanics (fees, slippage, and turbulence).
