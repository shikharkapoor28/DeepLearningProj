# UI Screens

## 1. Dashboard (Overview)
- **Components**: Sidebar Navigation, Global Metrics Cards (Total Profit, Active Agents, System Health), Recent Activity Feed.
- **Data Dependencies**: `GET /api/v1/health`, `GET /api/v1/models` (latest).
- **Interaction Model**: Read-only overview. Clicking an active agent routes to the Replay/Live mode.

## 2. Training View
- **Components**: Hyperparameter Form, Terminal Log Output, Live Loss Curves (Actor loss, Critic loss, Entropy), Pbar Progress.
- **Data Dependencies**: `POST /api/v1/experiments` (start), SSE/WebSockets for live logs and tensorboard-like curves.
- **Interaction Model**: User selects config overrides from dropdowns -> clicks "Start Training" -> UI switches to monitoring mode showing real-time loss graphs.

## 3. Experiment Comparison
- **Components**: Metric Data Table (sortable by Sharpe, Returns, etc.), Radar Charts for risk-return profile, Parallel Coordinates Plot for hyperparameters.
- **Data Dependencies**: `GET /api/v1/experiments` (batch fetch).
- **Interaction Model**: User checks boxes next to 2-5 experiments -> UI renders side-by-side radar charts and equity curves for visual comparison.

## 4. Replay Mode (Simulation Streaming)
- **Components**: 
  - Main Panel: Lightweight Charts Candlestick graph with buy/sell markers overlay.
  - Sidebar: Portfolio State widget (Cash vs Holdings donut chart).
  - Bottom Panel: Transaction Log table.
  - Top Bar: Play/Pause/Scrub controls, Speed multiplier (1x, 10x, 100x).
- **Data Dependencies**: `POST /api/v1/simulation/start`, `ws://.../simulation/{id}`.
- **Interaction Model**: User clicks Play -> WebSocket pipes data -> Charts update at 60fps. User can pause simulation to inspect specific trades. Provides an "Explanation" toggle to overlay attention heatmaps on the chart.
