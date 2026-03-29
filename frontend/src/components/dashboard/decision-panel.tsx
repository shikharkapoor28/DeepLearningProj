"use client";

import { TradeAction, Explanation } from "@/types/stream";
import { Panel } from "./panel";
import { BrainCircuit } from "lucide-react";
import { motion } from "framer-motion";

export function DecisionPanel({ action, explanation }: { action?: TradeAction, explanation?: Explanation }) {
  if (!action || !explanation) return <Panel title="Agent Decision">Waiting...</Panel>;

  const btcWeight = action.target_weights["BTC"] || 0;
  const usdtWeight = action.target_weights["USDT"] || (1 - btcWeight);
  
  const lastTrade = action.executed_trades[action.executed_trades.length - 1];
  
  let actionColor = "text-zinc-400";
  if (lastTrade?.side === "buy") actionColor = "text-green-400";
  if (lastTrade?.side === "sell") actionColor = "text-red-400";

  return (
    <Panel title="Agent Decision" className="h-[380px]">
      {/* Action Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex flex-col">
          <span className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Last Action</span>
          <span className={`text-2xl font-bold uppercase tracking-tight ${actionColor}`}>
            {lastTrade?.side || "HOLD"}
          </span>
        </div>
        <div className="h-10 w-[1px] bg-zinc-800" />
        <div className="flex flex-col items-end">
           <span className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Conviction</span>
           <span className="text-xl font-mono text-zinc-300">{(Math.max(btcWeight, usdtWeight) * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Target Allocation */}
      <div className="mb-4 text-xs">
        <div className="flex justify-between mb-1.5 text-zinc-400">
          <span>Target Allocation</span>
        </div>
        <div className="h-2 w-full bg-zinc-800 rounded-full flex overflow-hidden">
           <motion.div animate={{ width: `${btcWeight * 100}%` }} className="bg-blue-500 h-full" transition={{ type: "spring" }} />
           <motion.div animate={{ width: `${usdtWeight * 100}%` }} className="bg-zinc-600 h-full" transition={{ type: "spring" }} />
        </div>
        <div className="flex justify-between mt-1.5 text-zinc-500">
          <span>{Math.round(btcWeight * 100)}% BTC</span>
          <span>{Math.round(usdtWeight * 100)}% USDT</span>
        </div>
      </div>

      {/* Explainability Feature Bar Chart */}
      <div className="flex-1 mt-2 flex flex-col justify-end border-t border-zinc-800/50 pt-3">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase flex items-center gap-1.5 mb-3">
          <BrainCircuit className="w-3 h-3" />
          Feature Contributions
        </h4>
        
        <div className="space-y-2.5 overflow-y-auto max-h-[140px] pr-2">
          {explanation.top_features.map((feat, i) => {
            const weight = explanation.attention_weights[i] || 0.1;
            const percentage = (weight * 100).toFixed(1);
            
            return (
              <div key={i} className="flex flex-col text-xs">
                <div className="flex justify-between mb-1">
                  <span className="text-zinc-300 font-medium truncate pr-2">{feat.replace(/_/g, ' ').toUpperCase()}</span>
                  <span className="text-blue-400 font-mono">{percentage}%</span>
                </div>
                <div className="h-1 w-full bg-zinc-800/50 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }} 
                    animate={{ width: `${weight * 100}%` }} 
                    className="h-full bg-gradient-to-r from-blue-600 to-blue-400"
                    transition={{ type: "spring", stiffness: 50 }} 
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Panel>
  );
}
