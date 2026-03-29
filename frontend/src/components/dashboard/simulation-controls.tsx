"use client";

import { Panel } from "./panel";
import { Play, Square, StepBack, StepForward, RotateCcw } from "lucide-react";

interface ControlsProps {
  onStart: () => void;
  onStop: () => void;
  onReplayStep?: (step: number) => void;
  onExitReplay?: () => void;
  isConnected: boolean;
  isReplaying?: boolean;
}

export function SimulationControls({ onStart, onStop, onReplayStep, onExitReplay, isConnected, isReplaying }: ControlsProps) {
  return (
    <Panel title="Timeline Controls">
      <div className="flex gap-2 h-full items-center">
        {isReplaying ? (
           <button 
             onClick={onExitReplay} 
             className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 rounded-lg flex items-center justify-center gap-2 transition-colors"
           >
             <RotateCcw className="w-4 h-4" /> Live Feed
           </button>
        ) : (
           <button 
             onClick={onStart} 
             disabled={isConnected}
             className="flex-1 bg-white hover:bg-zinc-200 text-black font-semibold py-3 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
           >
             <Play className="w-4 h-4" /> Start Feed
           </button>
        )}
        
        <button 
          onClick={onStop}
          disabled={!isConnected && !isReplaying}
          className="px-4 bg-zinc-800 hover:bg-zinc-700 text-white font-semibold py-3 flex-1 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Square className="w-4 h-4" /> Stop
        </button>
        
        <div className="flex gap-1 ml-auto">
          <button 
            onClick={() => onReplayStep?.(-1)}
            title="Step Back"
            className={`px-4 bg-zinc-800 hover:bg-zinc-700 text-white py-3 rounded-lg transition-colors ${isReplaying ? 'ring-1 ring-blue-500/50 bg-zinc-700' : ''}`}
          >
            <StepBack className="w-4 h-4 text-zinc-300" />
          </button>
          <button 
            onClick={() => onReplayStep?.(1)}
            title="Step Forward"
            className={`px-4 bg-zinc-800 hover:bg-zinc-700 text-white py-3 rounded-lg transition-colors ${isReplaying ? 'ring-1 ring-blue-500/50 bg-zinc-700' : ''}`}
          >
            <StepForward className="w-4 h-4 text-zinc-300" />
          </button>
        </div>
      </div>
    </Panel>
  );
}
