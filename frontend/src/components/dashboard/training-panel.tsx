"use client";

import { Panel } from "./panel";
import { Activity } from "lucide-react";

export function TrainingPanel() {
  return (
    <Panel title="Training Health">
      <div className="flex-1 flex gap-6 mt-2">
        <div className="flex-1 flex flex-col justify-between">
          <div>
            <span className="text-xs text-zinc-500 uppercase font-semibold">Policy Loss</span>
            <div className="mt-1 flex items-end gap-2 text-zinc-300 font-mono">
              <span className="text-xl">-0.024</span>
              <span className="text-xs text-green-400 pb-1 flex items-center"><Activity className="w-3 h-3 mr-1"/> Stable</span>
            </div>
          </div>
          <div className="h-12 w-full bg-zinc-800/50 rounded flex items-end overflow-hidden p-1 gap-1">
             {Array.from({length: 20}).map((_, i) => (
                <div key={i} suppressHydrationWarning className="flex-1 bg-red-400/40 rounded-t-sm" style={{height: `${Math.random()*40 + 20}%`}} />
             ))}
          </div>
        </div>

        <div className="w-[1px] bg-zinc-800/50 my-2" />

        <div className="flex-1 flex flex-col justify-between">
          <div>
            <span className="text-xs text-zinc-500 uppercase font-semibold">Value Loss</span>
            <div className="mt-1 flex items-end gap-2 text-zinc-300 font-mono">
              <span className="text-xl">1.452</span>
              <span className="text-xs text-green-400 pb-1 flex items-center">Converged</span>
            </div>
          </div>
          <div className="h-12 w-full bg-zinc-800/50 rounded flex items-end overflow-hidden p-1 gap-1">
             {Array.from({length: 20}).map((_, i) => (
                <div key={i} suppressHydrationWarning className="flex-1 bg-green-400/40 rounded-t-sm" style={{height: `${Math.random()*10 + 10}%`}} />
             ))}
          </div>
        </div>
      </div>
    </Panel>
  );
}
