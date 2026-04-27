"use client";

import { useEffect, useState, useRef } from "react";
import { SimulationPayload } from "@/types/stream";
import { WebSocketClient } from "@/lib/websocket-client";

const MOCK_DATA = (): SimulationPayload => ({
  timestamp: Date.now(),
  state: { price: 710.0 + (Math.random() * 5 - 2.5), volume_1h: 1500000 + Math.random() * 50000, rsi: 45 + Math.random() * 20, macd: 0.5 + Math.random(), turbulence: 0.1 },
  action: { 
    target_weights: { "SPY": 0.6, "Cash": 0.4 }, 
    executed_trades: [{ asset: "SPY", side: ["buy", "sell", "hold"][Math.floor(Math.random() * 3)] as "buy" | "sell" | "hold", amount: 10, price: 710.0 }], 
    friction_cost: 0.54 
  },
  reward: (Math.random() - 0.4) * 0.05,
  portfolio: { total_value: 105000 + (Math.random()*100 - 50), cash: 40000, holdings: { "SPY": 65000 }, drawdown: 0.02 + Math.random()*0.01 },
  explanation: { attention_weights: [0.1, 0.2, 0.5, 0.1, 0.1], top_features: ["volume_spike", "rsi_oversold"] }
});

// React hook to manage the live streaming of trading simulation data
export function useSimulationStream(sessionId: string) {
  // State for the latest data tick
  const [data, setData] = useState<SimulationPayload | null>(null);
  // Maintain a rolling history of the last 100 ticks for plotting charts
  const [history, setHistory] = useState<SimulationPayload[]>([]);
  // Track connection status for UI indicators
  const [isConnected, setIsConnected] = useState(false);
  
  const clientRef = useRef<WebSocketClient | null>(null);
  const mockInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const wsUrl = `ws://localhost:8000/ws/simulation/${sessionId}`;
    const client = new WebSocketClient(wsUrl);
    clientRef.current = client;

    const unsubscribe = client.subscribe((payload) => {
      setData(payload);
      setHistory(prev => [...prev.slice(-99), payload]); // Keep last 100
      setIsConnected(true);
    });

    client.connect();

    // Fallback to mock data if the Python backend connection fails quickly.
    // This ensures UI development can continue unhindered even without the server.
    const checkConnection = setTimeout(() => {
      if (!client.isConnected) {
        console.warn("Using mock data stream...");
        mockInterval.current = setInterval(() => {
          const mock = MOCK_DATA();
          setData(mock);
          setHistory(prev => [...prev.slice(-99), mock]);
        }, 1000);
      }
    }, 2000);

    return () => {
      unsubscribe();
      client.disconnect();
      clearTimeout(checkConnection);
      if (mockInterval.current) clearInterval(mockInterval.current);
    };
  }, [sessionId]);

  const sendCommand = (cmd: string) => {
    clientRef.current?.send({ command: cmd });
  };

  return { data, history, isConnected, sendCommand };
}
