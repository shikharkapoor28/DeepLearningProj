"use client";

import { useState } from "react";
import { useSimulationStream } from "@/hooks/use-simulation-stream";
import { AppShell } from "@/components/layout/app-shell";
import { MarketView } from "@/components/dashboard/market-view";
import { DecisionPanel } from "@/components/dashboard/decision-panel";
import { PortfolioPanel } from "@/components/dashboard/portfolio-panel";
import { BenchmarkPanel } from "@/components/dashboard/benchmark-panel";
import { TrainingPanel } from "@/components/dashboard/training-panel";
import { SimulationControls } from "@/components/dashboard/simulation-controls";
import { EquityCurvePanel } from "@/components/dashboard/equity-curve";

export default function Dashboard() {
  const { data, history, isConnected, sendCommand } = useSimulationStream("demo_session");
  const [replayIndex, setReplayIndex] = useState<number | null>(null);

  // Task 3: Replay System Logic
  // If we are in replay mode, we slice the history and extract the specific focused data point to pass to the UI
  const isReplaying = replayIndex !== null;
  const displayHistory = isReplaying ? history.slice(0, replayIndex! + 1) : history;
  const displayData = displayHistory.length > 0 ? displayHistory[displayHistory.length - 1] : data;

  const handleReplayScrub = (step: number) => {
    if (history.length === 0) return;
    let newIndex = (replayIndex ?? history.length - 1) + step;
    newIndex = Math.max(0, Math.min(newIndex, history.length - 1));
    setReplayIndex(newIndex);
  };

  const handleExitReplay = () => setReplayIndex(null);

  return (
    <AppShell>
      <div className="grid grid-cols-12 gap-6">
        
        {/* Left Column: Market View & Equity */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">
          <MarketView history={displayHistory} />
          
          <EquityCurvePanel history={displayHistory} />
          
          {/* Bottom row under market view */}
          <div className="grid grid-cols-2 gap-6 mt-auto">
            <TrainingPanel />
            <SimulationControls 
              isConnected={isConnected} 
              isReplaying={isReplaying}
              onStart={() => { sendCommand("start"); handleExitReplay(); }} 
              onStop={() => sendCommand("stop")} 
              onReplayStep={(step) => handleReplayScrub(step)}
              onExitReplay={handleExitReplay}
            />
          </div>
        </div>

        {/* Right Column: Reasoning & Portfolio */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
          <DecisionPanel action={displayData?.action} explanation={displayData?.explanation} />
          <PortfolioPanel portfolio={displayData?.portfolio} />
          <BenchmarkPanel />
        </div>

      </div>
    </AppShell>
  );
}
