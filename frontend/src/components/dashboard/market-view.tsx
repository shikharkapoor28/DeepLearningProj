"use client";

import { MarketState, SimulationPayload } from "@/types/stream";
import { Panel } from "./panel";
import { CandlestickChart } from "./candlestick";
import { Activity } from "lucide-react";

export function MarketView({ history }: { history: SimulationPayload[] }) {
  if (!history || history.length === 0) return <Panel title="Market View" className="h-[400px]"><div className="flex items-center justify-center h-full text-zinc-500 animate-pulse">Waiting for stream...</div></Panel>;

  const latest = history[history.length - 1];
  const state = latest.state;
  const timeString = new Date(latest.timestamp).toLocaleTimeString();

  // Regime Detection Logic (Task 4)
  let regime = "Range Bound";
  let regimeColor = "bg-zinc-800 text-zinc-300";
  
  if (state.turbulence && state.turbulence > 0.5) {
      regime = "High Volatility";
      regimeColor = "bg-orange-950/50 text-orange-400 border border-orange-900/50";
  } else if (state.rsi && state.rsi > 60) {
      regime = "Bull Trend";
      regimeColor = "bg-green-950/50 text-green-400 border border-green-900/50";
  } else if (state.rsi && state.rsi < 40) {
      regime = "Bear Trend";
      regimeColor = "bg-red-950/50 text-red-400 border border-red-900/50";
  }

  return (
    <Panel title="Market View & OHLC" className="h-[450px]">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-3xl font-bold text-white tracking-tight">${state.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</h3>
          <div className="flex items-center gap-3 mt-1">
            <p className="text-zinc-400 text-sm">SPY ETF | Last update: {timeString}</p>
            <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-widest ${regimeColor}`}>
              Regime: {regime}
            </span>
          </div>
        </div>
        <div className="flex gap-4">
           <div className="bg-zinc-800/50 px-3 py-1.5 rounded-md flex flex-col items-end">
             <span className="text-xs text-zinc-500 uppercase font-semibold tracking-wider">RSI</span>
             <span className={`font-mono ${!state.rsi ? 'text-zinc-400' : state.rsi > 70 ? 'text-red-400' : state.rsi < 30 ? 'text-green-400' : 'text-zinc-300'}`}>{state.rsi?.toFixed(1) || '--'}</span>
           </div>
           <div className="bg-zinc-800/50 px-3 py-1.5 rounded-md flex flex-col items-end">
             <span className="text-xs text-zinc-500 uppercase font-semibold tracking-wider">Turbulence</span>
             <span className="font-mono text-zinc-300">{state.turbulence?.toFixed(3) || '--'}</span>
           </div>
        </div>
      </div>
      
      {/* Real Candlestick Chart */}
      <div className="flex-1 w-full relative">
        <CandlestickChart history={history} />
      </div>
    </Panel>
  );
}
