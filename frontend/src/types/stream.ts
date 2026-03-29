export interface MarketState {
  price: number;
  volume_1h: number;
  rsi?: number;
  macd?: number;
  turbulence?: number;
}

export interface TradeAction {
  target_weights: Record<string, number>;
  executed_trades: Array<{
    asset: string;
    side: "buy" | "sell" | "hold";
    amount: number;
    price: number;
  }>;
  friction_cost: number;
}

export interface Portfolio {
  total_value: number;
  cash: number;
  holdings: Record<string, number>;
  drawdown: number;
}

export interface Explanation {
  attention_weights: number[];
  top_features: string[];
}

export interface SimulationPayload {
  timestamp: number;
  state: MarketState;
  action: TradeAction;
  reward: number;
  portfolio: Portfolio;
  explanation: Explanation;
}
