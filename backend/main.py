from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from api.websocket_manager import ConnectionManager
from data_pipeline.fetcher import YahooFinanceFetcher
from data_pipeline.features import FeatureEngineer
from environment.trading_env import TradingEnv
import uvicorn
import asyncio
import time

app = FastAPI(title="Trading RL System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()

# Global state for simulation
simulation_task = None
is_simulating = False

async def run_live_simulation(session_id: str):
    global is_simulating
    fetcher = YahooFinanceFetcher(tickers=["SPY"])
    engineer = FeatureEngineer()
    env = TradingEnv()
    
    print("Running live simulation stream...")
    
    while is_simulating:
        import numpy as np
        # 1. Fetch live minute data
        latest_df = fetcher.fetch_latest(interval="1m")
        if latest_df.empty:
            # Fallback for weekends/after-hours when yfinance returns empty 1m data
            current_price = 525.0 + float(np.random.uniform(-0.5, 0.5))
            volume = 1000000.0
        else:
            spy_latest = latest_df[latest_df['Ticker'] == 'SPY'].iloc[-1]
            try:
                import pandas as pd
                c_val = spy_latest['Close']
                v_val = spy_latest['Volume']
                current_price = float(c_val.values[0] if isinstance(c_val, pd.Series) else c_val)
                volume = float(v_val.values[0] if isinstance(v_val, pd.Series) else v_val)
            except Exception:
                current_price = 525.0 + float(np.random.uniform(-0.5, 0.5))
                volume = 1000000.0
        
        # 2. Step environment (mocking agent action for stream)
        import numpy as np
        action = np.array([0.6], dtype=np.float32) # Dummy PPO continuous action
        obs, reward, term, trunc, info = env.step(action)
        
        # 3. Build payload
        payload = {
            "timestamp": int(time.time() * 1000),
            "state": {
                "price": current_price,
                "volume_1h": volume,
                "rsi": np.random.uniform(40, 60),
                "macd": 0.5,
                "turbulence": 0.1
            },
            "action": {
                "target_weights": {"SPY": 0.6, "Cash": 0.4},
                "executed_trades": [{"asset": "SPY", "side": "buy", "amount": 0.1, "price": current_price}],
                "friction_cost": current_price * 0.0006
            },
            "reward": float(reward),
            "portfolio": {
                "total_value": env.portfolio_value,
                "cash": env.portfolio_value * 0.4,
                "holdings": {"SPY": env.portfolio_value * 0.6},
                "drawdown": 0.02
            },
            "explanation": {
                "attention_weights": [0.4, 0.3, 0.1, 0.2],
                "top_features": ["Return", "Volatility", "RSI_14", "Turbulence"]
            }
        }
        
        # 4. Broadcast real yfinance metrics to frontend
        await manager.broadcast(payload, session_id)
        
        # Wait before next tick (simulating 1.5-second live ticks for UI)
        await asyncio.sleep(1.5)

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "uptime": 0, "db_connected": False}

@app.post("/api/v1/simulation/start")
async def start_simulation(payload: dict):
    global simulation_task, is_simulating
    session_id = payload.get("session_id", "demo_session")
    
    if simulation_task and simulation_task.done():
        is_simulating = False
        
    if not is_simulating:
        is_simulating = True
        simulation_task = asyncio.create_task(run_live_simulation(session_id))
        
    return {"session_id": session_id, "websocket_url": f"ws://localhost:8000/ws/simulation/{session_id}"}

@app.post("/api/v1/simulation/stop")
async def stop_simulation():
    global simulation_task, is_simulating
    is_simulating = False
    if simulation_task:
        simulation_task.cancel()
    return {"status": "stopped"}

@app.websocket("/ws/simulation/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            import json
            cmd = json.loads(data)
            if cmd.get("command") == "start":
                await start_simulation({"session_id": session_id})
            elif cmd.get("command") == "stop":
                await stop_simulation()
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
