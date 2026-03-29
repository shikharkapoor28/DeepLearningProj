"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, IChartApi, ISeriesApi, LineSeries } from "lightweight-charts";
import { SimulationPayload } from "@/types/stream";
import { Panel } from "./panel";

export function EquityCurvePanel({ history }: { history: SimulationPayload[] }) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const rlSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bhSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#a1a1aa",
      },
      grid: {
        vertLines: { color: "rgba(39, 39, 42, 0.5)" }, // zinc-800
        horzLines: { color: "rgba(39, 39, 42, 0.5)" },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // RL Agent PnL Line
    rlSeriesRef.current = chart.addSeries(LineSeries, {
      color: "#3b82f6", // blue-500
      lineWidth: 2,
      crosshairMarkerVisible: true,
      lastValueVisible: true,
      priceLineVisible: false,
    });

    // Buy & Hold Baseline PnL Line
    bhSeriesRef.current = chart.addSeries(LineSeries, {
      color: "#fb923c", // orange-400
      lineWidth: 2,
      lineStyle: 2, // Dashed
      crosshairMarkerVisible: false,
      lastValueVisible: true,
      priceLineVisible: false,
    });

    chartRef.current = chart;

    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current?.clientWidth });
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!rlSeriesRef.current || !bhSeriesRef.current || history.length === 0) return;

    const startValue = 100000;
    const initialPrice = history[0].state.price;

    const rlData = [];
    const bhData = [];

    for (const record of history) {
      const time = (record.timestamp / 1000) as import("lightweight-charts").Time;
      
      // RL PnL
      rlData.push({ time, value: record.portfolio.total_value });

      // Simulate B&H: 100k fully allocated at start price
      const shares = startValue / initialPrice;
      const bhValue = shares * record.state.price;
      bhData.push({ time, value: bhValue });
    }

    const uniqueRlData = Array.from(new Map(rlData.map(item => [item.time, item])).values());
    const uniqueBhData = Array.from(new Map(bhData.map(item => [item.time, item])).values());

    rlSeriesRef.current.setData(uniqueRlData);
    bhSeriesRef.current.setData(uniqueBhData);

  }, [history]);

  return (
    <Panel title="Equity Curve (Portfolio vs Baseline)" className="h-[300px]">
      <div className="flex justify-between items-center mb-2 px-1 text-xs">
        <div className="flex gap-4">
          <div className="flex items-center gap-1.5 text-blue-400">
            <div className="w-2 h-2 rounded-full bg-blue-500" /> RL Agent
          </div>
          <div className="flex items-center gap-1.5 text-orange-400">
            <div className="w-2 h-2 rounded-full bg-orange-400" /> Baseline (Buy&Hold)
          </div>
        </div>
      </div>
      <div ref={chartContainerRef} className="flex-1 w-full" />
    </Panel>
  );
}
