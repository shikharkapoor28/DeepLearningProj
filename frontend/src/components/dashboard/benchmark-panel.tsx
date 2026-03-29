"use client";

import { Panel } from "./panel";

export function BenchmarkPanel() {
  return (
    <Panel title="Live Benchmarks">
      <div className="w-full h-full overflow-hidden flex flex-col justify-center">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-zinc-500">
              <th className="pb-2 font-medium">Strategy</th>
              <th className="pb-2 font-medium text-right">Return</th>
              <th className="pb-2 font-medium text-right">Sharpe</th>
              <th className="pb-2 font-medium text-right">Drawdown</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            <tr className="text-zinc-200">
              <td className="py-3 font-semibold text-blue-400">PPO Agent</td>
              <td className="py-3 text-right text-green-400">+45.2%</td>
              <td className="py-3 text-right font-mono">2.41</td>
              <td className="py-3 text-right text-red-400">15.1%</td>
            </tr>
            <tr className="text-zinc-400">
              <td className="py-3">Buy & Hold</td>
              <td className="py-3 text-right text-green-400/70">+20.5%</td>
              <td className="py-3 text-right font-mono">1.12</td>
              <td className="py-3 text-right text-red-400/70">35.4%</td>
            </tr>
            <tr className="text-zinc-500">
              <td className="py-3">Random</td>
              <td className="py-3 text-right text-red-500/70">-10.2%</td>
              <td className="py-3 text-right font-mono">0.05</td>
              <td className="py-3 text-right text-red-500/70">42.8%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
