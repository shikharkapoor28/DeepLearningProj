"""
Main Application Entrypoint.

This module is responsible for running the FastAPI application and orchestrating
the core live trading simulation. It ties together the data pipeline, the RL
trading environment, the trained policy, and the websocket communication with the frontend.
"""

# Standard Library Imports
import asyncio
import math
import os
import time
from typing import Dict, Optional, Tuple

# Third-Party Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import uvicorn

# Local Application Imports
from api.websocket_manager import ConnectionManager
from broker.base import Order
from broker.paper import PaperBroker, PaperBrokerConfig
from data_pipeline.feature_utils import (
    build_feature_matrix,
    compute_features,
    flatten_yfinance_columns,
    json_safe,
)
from data_pipeline.fetcher import YahooFinanceFetcher
from environment.trading_env import TradingEnv
from rl_core.explainability import ExplainabilityLayer
from rl_core.trainer import get_transformer_policy_kwargs

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

def _rule_action_from_rsi(rsi: float, current_weight: float) -> float:
    """
    Simple baseline so portfolio/reward moves realistically in the demo.
    - Oversold: go risk-on (100%)
    - Overbought: go risk-off (0%)
    - Neutral: hold a constant balanced allocation (60%)
    """
    if isinstance(rsi, float) and not math.isfinite(rsi):
        return 0.6
    if rsi <= 30:
        return 1.0
    if rsi >= 70:
        return 0.0
    return 0.6

async def run_live_simulation(session_id: str):
    global is_simulating
    fetcher = YahooFinanceFetcher(tickers=["SPY"])
    env: Optional[TradingEnv] = None
    policy = None
    explainer: Optional[ExplainabilityLayer] = None

    # Execution mode (paper broker). Default off (env simulation only).
    broker_mode = os.getenv("BROKER_MODE", "env").lower()  # env | paper
    broker = None
    if broker_mode == "paper":
        broker = PaperBroker(
            PaperBrokerConfig(
                starting_cash=float(os.getenv("BROKER_START_CASH", "100000")),
                fee_bps=float(os.getenv("BROKER_FEE_BPS", "6")),
                slippage_bps=float(os.getenv("BROKER_SLIPPAGE_BPS", "2")),
            )
        )
        print("Paper broker enabled (BROKER_MODE=paper).")

    # Optional: load a trained PPO policy (Stable-Baselines3)
    model_path = os.getenv("PPO_MODEL_PATH")
    if model_path:
        try:
            from stable_baselines3 import PPO
            policy = PPO.load(
                model_path,
                custom_objects={"policy_kwargs": get_transformer_policy_kwargs()},
            )
            explainer = ExplainabilityLayer(policy)
            print(f"Loaded PPO model from {model_path}")
        except Exception as e:
            print(f"Failed to load PPO model at {model_path}: {e}")

    # Rolling OHLCV history for indicators / env features
    hist: pd.DataFrame = pd.DataFrame()
    last_ts = None
    seeded = False
    
    print("Running live simulation stream...")
    
    while is_simulating:
        # Seed history once so indicators/env work immediately (otherwise you'd need ~100+ minutes).
        if not seeded:
            try:
                seed_df = fetcher.fetch_recent(period="1d", interval="1m")
                seed_df = flatten_yfinance_columns(seed_df)
                if not seed_df.empty:
                    seed_df = seed_df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
                    hist = pd.concat([hist, seed_df], axis=0).tail(500)
                    if not hist.empty:
                        last_ts = hist.index[-1]
                seeded = True
            except Exception:
                seeded = True

        # 1) Fetch latest bar and maintain rolling OHLCV window
        latest_df = fetcher.fetch_latest(interval="1m")
        latest_df = flatten_yfinance_columns(latest_df)

        if not latest_df.empty:
            row = latest_df.iloc[[-1]][["Open", "High", "Low", "Close", "Volume"]].copy()
            ts = row.index[-1]

            if last_ts is None or ts != last_ts:
                last_ts = ts
                hist = pd.concat([hist, row], axis=0)
                hist = hist.tail(500)  # cap memory

        # 2) Compute real indicators from history (requires enough bars)
        # Need: volatility(20) and turbulence(100) => at least ~120 bars to be stable.
        # We fall back gracefully if there's not enough yet.
        # volume_1h: rolling sum of last 60 1m bars if available
        volume_1h = float(hist["Volume"].tail(60).sum()) if len(hist) >= 1 else 1_000_000.0

        latest_state = {
            "price": float(hist["Close"].iloc[-1]) if not hist.empty else 525.0,
            "volume_1h": volume_1h,
            "rsi": float("nan"),
            "macd": float("nan"),
            "turbulence": float("nan"),
        }

        feature_matrix = None
        close_prices = None
        try:
            feat_df = compute_features(hist)
            feature_matrix, close_prices, latest_state = build_feature_matrix(feat_df)
        except Exception:
            # Not enough data yet (or transient fetch issues). Keep streaming price/volume anyway.
            pass

        # 3) Use TradingEnv real-data mode once we have enough features
        reward = 0.0
        target_w = 0.6
        executed_trade = {"asset": "SPY", "side": "hold", "amount": 0.0, "price": float(latest_state["price"])}
        friction_cost = float(latest_state["price"]) * 0.0006
        portfolio_total = 100000.0
        portfolio_cash = 100000.0
        portfolio_holdings_value = 0.0
        portfolio_drawdown = 0.0

        if feature_matrix is not None and close_prices is not None:
            if env is None:
                env = TradingEnv(
                    config={
                        "seq_len": 64,
                        "initial_balance": 100000.0,
                        "trading_fee_bps": 6,
                        "slippage_bps": 2,
                        "reward_scaling": 100.0,
                        "max_drawdown_stop": 0.30,
                    },
                    feature_matrix=feature_matrix,
                    close_prices=close_prices,
                )
                env.reset()
            else:
                # Update env's live data buffers without resetting portfolio
                env._feature_matrix = feature_matrix  # type: ignore[attr-defined]
                env._close_prices = close_prices      # type: ignore[attr-defined]

            current_w = float(getattr(env, "weight", 0.0))
            # Align env index to the newest complete return (len-2 -> len-1)
            try:
                env._data_idx = max(env.seq_len - 1, len(close_prices) - 2)  # type: ignore[attr-defined]
                env.state = env._observation_at(env._data_idx)  # type: ignore[attr-defined]
            except Exception:
                pass

            if policy is not None:
                try:
                    pred, _ = policy.predict(env.state, deterministic=True)
                    target_w = float(np.clip(np.asarray(pred, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
                except Exception:
                    target_w = _rule_action_from_rsi(latest_state["rsi"], current_w)
            else:
                target_w = _rule_action_from_rsi(latest_state["rsi"], current_w)

            if broker is None:
                # Env-only simulation
                _, reward, _, _, _ = env.step(np.array([target_w], dtype=np.float32))

                try:
                    side = "buy" if float(target_w) > float(current_w) else ("sell" if float(target_w) < float(current_w) else "hold")
                except Exception:
                    side = "hold"
                executed_trade = {
                    "asset": "SPY",
                    "side": side,
                    "amount": float(abs(float(target_w) - float(current_w))),
                    "price": float(latest_state["price"]),
                }
                try:
                    turnover = abs(float(target_w) - float(current_w))
                    friction_cost = float(env.portfolio_value) * turnover * float(env.trading_fee + env.slippage)
                except Exception:
                    pass

                portfolio_total = float(env.portfolio_value)
                w = float(getattr(env, "weight", target_w))
                portfolio_cash = float(env.portfolio_value * (1.0 - w))
                portfolio_holdings_value = float(env.portfolio_value * w)
                portfolio_drawdown = float(
                    1.0 - float(env.portfolio_value) / (float(getattr(env, "peak_value", env.portfolio_value)) + 1e-12)
                )
            else:
                # Paper broker execution: convert target weight -> order qty
                ts_ms = int(time.time() * 1000)
                px = float(latest_state["price"])
                acct = broker.get_account({"SPY": px})
                equity = float(acct.equity)
                cur_qty = float(acct.positions.get("SPY", 0.0))
                cur_value = cur_qty * px
                cur_w_broker = (cur_value / equity) if equity > 0 else 0.0

                target_value = float(target_w) * equity
                delta_value = target_value - cur_value
                min_trade_value = float(os.getenv("MIN_TRADE_VALUE", "10"))  # dollars

                if abs(delta_value) >= min_trade_value and px > 0:
                    if delta_value > 0:
                        side = "buy"
                        qty = delta_value / px
                    else:
                        side = "sell"
                        qty = (-delta_value) / px

                    fill = broker.place_order(
                        Order(symbol="SPY", side=side, qty=float(qty)),
                        mark_price=px,
                        timestamp_ms=ts_ms,
                    )
                    executed_trade = {
                        "asset": fill.symbol,
                        "side": fill.side,
                        "amount": float(fill.qty),
                        "price": float(fill.price),
                    }
                    friction_cost = float(fill.fee)
                else:
                    executed_trade = {"asset": "SPY", "side": "hold", "amount": 0.0, "price": px}
                    friction_cost = 0.0

                acct2 = broker.get_account({"SPY": px})
                portfolio_total = float(acct2.equity)
                portfolio_cash = float(acct2.cash)
                portfolio_holdings_value = float(acct2.positions.get("SPY", 0.0) * px)
                portfolio_drawdown = 0.0

                # Keep env stepping so reward/obs still make sense for streaming (but portfolio comes from broker)
                _, reward, _, _, _ = env.step(np.array([target_w], dtype=np.float32))
        
        # 3. Build explanation from real explainability layer
        explanation_payload = {"attention_weights": None, "top_features": []}
        if explainer is not None and env is not None and env.state is not None:
            try:
                attn = explainer.get_attention_weights(env.state)
                if attn is not None:
                    # Send average attention over the last row (most recent timestep)
                    explanation_payload["attention_weights"] = attn[-1].tolist()
                explanation_payload["top_features"] = explainer.get_top_features(env.state)
            except Exception:
                pass

        # 4. Build payload
        payload = {
            "timestamp": int(time.time() * 1000),
            "state": {
                "price": latest_state["price"],
                "volume_1h": latest_state["volume_1h"],
                "rsi": latest_state["rsi"],
                "macd": latest_state["macd"],
                "turbulence": latest_state["turbulence"],
            },
            "action": {
                "target_weights": {"SPY": float(target_w), "Cash": float(1.0 - target_w)},
                "executed_trades": [executed_trade],
                "friction_cost": float(friction_cost),
            },
            "reward": float(reward),
            "portfolio": {
                "total_value": float(portfolio_total),
                "cash": float(portfolio_cash),
                "holdings": {"SPY": float(portfolio_holdings_value)},
                "drawdown": float(portfolio_drawdown),
            },
            "explanation": explanation_payload,
        }
        
        # 5. Broadcast real yfinance metrics to frontend
        await manager.broadcast(json_safe(payload), session_id)
        
        # Wait before next tick (simulating 1.5-second live ticks for UI)
        await asyncio.sleep(1.5)

@app.get("/api/v1/health")
async def health_check():
    """
    Basic health check endpoint. Useful for monitoring the service uptime.
    """
    return {"status": "ok", "uptime": 0, "db_connected": False}

@app.post("/api/v1/simulation/start")
async def start_simulation(payload: dict):
    """
    Kicks off the live trading simulation as a background task.
    If a simulation is already running, we restart it.
    """
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
    """
    Gracefully stops the running live simulation.
    """
    global simulation_task, is_simulating
    is_simulating = False
    if simulation_task:
        simulation_task.cancel()
    return {"status": "stopped"}

@app.websocket("/ws/simulation/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Websocket endpoint for streaming live data metrics and RL agent actions 
    directly to the frontend dashboard.
    """
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
