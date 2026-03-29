"use client";

import { useEffect, useState, useRef } from "react";
import { SimulationPayload } from "@/types/stream";
import { WebSocketClient } from "@/lib/websocket-client";

const MOCK_DATA = (): SimulationPayload => ({
  timestamp: Date.now(),
  state: { price: 65000 + (Math.random() * 1000 - 500), volume_1h: 1500 + Math.random() * 500, rsi: 45 + Math.random() * 20, macd: 0.5 + Math.random(), turbulence: 0.1 },
  action: { 
    target_weights: { "BTC": 0.6, "USDT": 0.4 }, 
    executed_trades: [{ asset: "BTC", side: Math.random() > 0.5 ? "buy" : "hold", amount: 0.1, price: 65000 }], 
    friction_cost: 6.54 
  },
  reward: (Math.random() - 0.4) * 0.05,
  portfolio: { total_value: 105000 + (Math.random()*100 - 50), cash: 40000, holdings: { "BTC": 65000 }, drawdown: 0.02 + Math.random()*0.01 },
  explanation: { attention_weights: [0.1, 0.2, 0.5, 0.1, 0.1], top_features: ["volume_spike", "rsi_oversold"] }
});

export function useSimulationStream(sessionId: string) {
  const [data, setData] = useState<SimulationPayload | null>(null);
  const [history, setHistory] = useState<SimulationPayload[]>([]);
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

    // Fallback to mock data if connection fails quickly (for UI building)
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
