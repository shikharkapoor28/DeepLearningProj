# API Contract

## REST Endpoints (FastAPI)

### Health
- `GET /api/v1/health`
  - **Description**: Returns system health and status.
  - **Response**: `{"status": "ok", "uptime": 3600, "db_connected": true}`

### Data
- `GET /api/v1/data/sources`
  - **Description**: List available historical datasets and live feeds.
- `POST /api/v1/data/ingest`
  - **Description**: Trigger a new data ingestion pipeline job.

### Simulation Control
- `POST /api/v1/simulation/start`
  - **Description**: Starts a replay or live simulation of an agent.
  - **Payload**: `{"model_id": "ppo_transformer_v1", "dataset": "eth_usdt_2023", "mode": "replay"}`
  - **Response**: `{"session_id": "sim_12345", "websocket_url": "ws://localhost:8000/ws/sim_12345"}`
- `POST /api/v1/simulation/stop/{session_id}`
  - **Description**: Terminates an active simulation.

### Model Listing
- `GET /api/v1/models`
  - **Description**: Lists all trained agents and checkpoints.
  - **Response**: Array of `{"id": "...", "algo": "PPO", "metrics": {...}, "created_at": "..."}`

### Experiments
- `GET /api/v1/experiments`
  - **Description**: List hyperparameter sweeps and ablations.
- `GET /api/v1/experiments/{id}/reports`
  - **Description**: Get full evaluation reports for a specific experiment.

---

## WebSocket (Live Streaming)

### Endpoint: `ws://<host>/ws/simulation/{session_id}`
- **Description**: Streams high-frequency updates from the environment back to the client.

### Payload Structure (JSON)
Every message from the server should conform to:
```json
{
  "timestamp": 1711468200000,
  "state": {
    "price": 65432.10,
    "volume_1h": 1500.5
  },
  "action": {
    "target_weights": {"BTC": 0.6, "USDT": 0.4},
    "executed_trades": [{"asset": "BTC", "side": "buy", "amount": 0.1, "price": 65433.00}],
    "friction_cost": 6.54
  },
  "reward": -0.012,
  "portfolio": {
    "total_value": 105000.00,
    "cash": 40000.00,
    "holdings": {"BTC": 65000.00},
    "drawdown": 0.02
  },
  "explanation": {
    "attention_weights": [0.1, 0.05, 0.8, 0.05],
    "top_features": ["volume_spike", "rsi_oversold"]
  }
}
```
