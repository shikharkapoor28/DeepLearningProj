# System Flow

The system operates continuously in a sequence combining training time, evaluation, and live-streaming. The core RL loop is designed around standard MDP principles but scoped for financial time series and streaming updates.

## 1. The Core RL Loop (Training & Execution)
1. **State**: The `data_pipeline` emits the current market state $S_t$ and feature embeddings. The `environment` concatenates the current portfolio allocation to form the full observation.
2. **Agent**: The observation is passed into the `rl_core`. The Transformer encoder extracts temporal context, and the Actor network outputs an **Action** $A_t$ (e.g., target asset weights). The Critic outputs a value estimate $V(S_t)$.
3. **Action → Environment**: The `environment` executes $A_t$, applying simulated market friction (slippage, commissions) to transition to state $S_{t+1}$.
4. **Reward**: The `environment` calculates the **Reward** $R_t$ (e.g., risk-adjusted return or differential Sharpe) and updates the portfolio.
5. **Experience Buffer**: The tuple $(S_t, A_t, R_t, S_{t+1})$ is appended to the PPO rollout buffer.
6. **PPO Update**: Once the buffer is full, the agent calculates advantages and updates the Transformer, Actor, and Critic weights using PPO's clipped surrogate objective. 

## 2. Evaluation Loop
- Runs asynchronously or periodically after epochs.
- Triggers a **Walk-Forward Validation**: The freshly updated agent trades on a held-out validation set.
- Calculates financial metrics (Max Drawdown, Win Rate, Sortino).
- Logs results to the `research` tracking system (e.g., W&B) and saves checkpoints.

## 3. Streaming to Frontend
- During a live simulation (or replay mode), the `api` module subscribes to the `environment` state and `explainability` metrics.
- As the loop progresses, $(S_t, Portfolio\_Value_t, Action_t, Attention\_Weights_t)$ are emitted over **WebSockets** at a fixed throttle rate (e.g., 100ms).
- The `frontend` receives these packets, updates internal `state_management`, and re-renders the `chart_system` and `explanation_ui` locally.
