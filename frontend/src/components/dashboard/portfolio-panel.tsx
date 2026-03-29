"use client";

import { Portfolio } from "@/types/stream";
import { Panel } from "./panel";
import { Wallet, PieChart } from "lucide-react";

export function PortfolioPanel({ portfolio }: { portfolio?: Portfolio }) {
  if (!portfolio) return <Panel title="Portfolio State">Waiting...</Panel>;

  const startValue = 100000; // Assuming 100k starting for demo
  const pnl = portfolio.total_value - startValue;
  const pnlPercent = (pnl / startValue) * 100;
  
  const pnlColor = pnl >= 0 ? "text-green-400" : "text-red-400";

  return (
    <Panel title="Portfolio State" className="h-[300px]">
      <div className="mb-6">
        <h3 className="text-3xl font-bold text-white tracking-tight">${portfolio.total_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</h3>
        <p className={`text-sm ${pnlColor} font-medium flex items-center gap-1 mt-1`}>
          {pnl >= 0 ? "+" : ""}{pnl.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} 
          ({pnlPercent >= 0 ? "+" : ""}{pnlPercent.toFixed(2)}%)
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 flex-1">
        <div className="bg-zinc-800/30 rounded-lg p-3 border border-zinc-800/50 flex flex-col justify-center">
          <span className="text-xs text-zinc-500 uppercase font-semibold mb-1 flex items-center gap-1.5"><Wallet className="w-3 h-3"/> Cash Balance</span>
          <span className="text-lg font-mono text-zinc-300">${portfolio.cash.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
        </div>
        
        <div className="bg-zinc-800/30 rounded-lg p-3 border border-zinc-800/50 flex flex-col justify-center">
          <span className="text-xs text-zinc-500 uppercase font-semibold mb-1 flex items-center gap-1.5"><PieChart className="w-3 h-3"/> Asset Value</span>
          <span className="text-lg font-mono text-zinc-300">${(portfolio.total_value - portfolio.cash).toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
        </div>

        <div className="col-span-2 bg-red-950/10 rounded-lg p-3 border border-red-900/20 flex items-center justify-between">
          <span className="text-xs text-red-400/70 uppercase font-semibold">Max Drawdown</span>
          <span className="text-sm font-mono text-red-400">{(portfolio.drawdown * 100).toFixed(2)}%</span>
        </div>
      </div>
    </Panel>
  );
}
