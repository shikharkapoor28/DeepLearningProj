# Architecture Responsibilities

## Backend

### `data_pipeline`
- **Purpose**: Ingest, clean, and pre-process market data from historical sources or live streams.
- **Inputs**: Exchange API streams, historical CSV files, SQL database feeds.
- **Outputs**: Normalized Pandas DataFrames, continuous streaming feature vectors.
- **Dependencies**: `pandas`, `numpy`, `ccxt`, `sqlalchemy`.

### `environment` (Gymnasium-style)
- **Purpose**: Simulate market dynamics and trading execution to train the RL agent. Manage portfolio state and transaction costs.
- **Inputs**: Actions from the RL agent (e.g., target portfolio weights or discrete trade sizes), state features from `data_pipeline`.
- **Outputs**: Observations (state), Rewards, Done flags, Info dicts (portfolio value, friction costs).
- **Dependencies**: `gymnasium`, `numpy`.

### `rl_core` (PPO + Actor-Critic + Transformer)
- **Purpose**: Learn optimal trading policies. Uses a Transformer encoder for sequence processing and PPO actor-critic heads for decision making.
- **Inputs**: Sequential state observations from the `environment`.
- **Outputs**: Action distributions (Actor), state value estimates (Critic).
- **Dependencies**: `stable-baselines3`, `torch`, `transformers`.

### `explainability`
- **Purpose**: Interpret the RL agent's decisions to build trust. Provide attribution for why specific trades were made.
- **Inputs**: Agent states, actions, and network weights/activations.
- **Outputs**: Feature importance scores, attention rollouts, SHAP values.
- **Dependencies**: `captum`, `shap`.

### `evaluation` (Metrics + Backtesting + Walk-Forward)
- **Purpose**: Rigorously assess strategy performance across different timeframes and regimes. Ensure avoidance of overfitting.
- **Inputs**: Action histories, portfolio values, market prices.
- **Outputs**: Finance-aware metrics (Sharpe, Sortino, Max Drawdown), backtest reports, walk-forward validation logs.
- **Dependencies**: `quantstats`, `empyrical`.

### `research` (Experiments + Ablations)
- **Purpose**: Facilitate model improvements through systematic experimentation and tracking.
- **Inputs**: Experiment configs, training logs, evaluation metrics.
- **Outputs**: Weight & Biases (W&B) runs, hyperparameter sweeps, ablation study reports.
- **Dependencies**: `wandb`, `optuna`.

### `api` (REST + WebSocket)
- **Purpose**: Serve the trained models, manage experiments, and stream live simulations to the frontend.
- **Inputs**: REST requests (Start experiment, get models), WebSocket subscriptions.
- **Outputs**: JSON responses, binary/JSON streaming data (prices, portfolio state).
- **Dependencies**: `fastapi`, `uvicorn`, `websockets`.

## Frontend

### `dashboard_layout`
- **Purpose**: Provide the main UI shell, navigation, and persistent state containers.
- **Inputs**: User interactions, routing states.
- **Outputs**: Rendered page structure.
- **Dependencies**: Next.js App Router, Tailwind CSS, shadcn/ui.

### `simulation_streaming`
- **Purpose**: Connect to the backend WebSocket and manage the incoming stream of live trading data.
- **Inputs**: Backend WebSocket feed.
- **Outputs**: State updates dispatched to the store.
- **Dependencies**: native WebSockets or `socket.io-client`.

### `state_management`
- **Purpose**: Hold the client-side state of the application centrally (e.g., active models, current portfolio value).
- **Inputs**: Actions from UI components and formatting streams.
- **Outputs**: Reactive state variables mapped to components.
- **Dependencies**: `zustand` or React Context.

### `chart_system`
- **Purpose**: Render high-performance financial charts (candlesticks, equity curves, volume).
- **Inputs**: Time-series arrays (prices, portfolio balance).
- **Outputs**: Interactive Canvas/SVG charts.
- **Dependencies**: `lightweight-charts` or `recharts`.

### `explanation_ui`
- **Purpose**: Visualize the explainability outputs from the backend (e.g., attention heatmaps on price charts).
- **Inputs**: Attention weights, SHAP vectors.
- **Outputs**: Heatmaps, feature importance bar charts.
- **Dependencies**: Custom D3.js or Canvas overlays.
