"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickSeries } from "lightweight-charts";
import { SimulationPayload } from "@/types/stream";

export function CandlestickChart({ history }: { history: SimulationPayload[] }) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#a1a1aa", // zinc-400
      },
      grid: {
        vertLines: { color: "#27272a" }, // zinc-800
        horzLines: { color: "#27272a" }, // zinc-800
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#4ade80", // green-400
      downColor: "#f87171", // red-400
      borderVisible: false,
      wickUpColor: "#4ade80",
      wickDownColor: "#f87171",
    });

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

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
    if (!seriesRef.current || history.length === 0) return;

    // Convert payload history to OHLC format
    // Since we only stream close prices locally, we mock OHLC based on the single price point stream 
    // for this visualization task. In production, the backend streams actual OHLC bars.
    const data = history.map((record, index) => {
      const prevPrice = index > 0 ? history[index - 1].state.price : record.state.price;
      const isUp = record.state.price >= prevPrice;
      const volatility = record.state.price * 0.002; // 0.2% mock volatility

      return {
        time: (record.timestamp / 1000) as import("lightweight-charts").Time,
        open: prevPrice,
        high: Math.max(prevPrice, record.state.price) + Math.random() * volatility,
        low: Math.min(prevPrice, record.state.price) - Math.random() * volatility,
        close: record.state.price,
      };
    });

    // Remove duplicates based on timestamp for lightweight-charts
    const uniqueData = Array.from(new Map(data.map(item => [item.time, item])).values());
    seriesRef.current.setData(uniqueData);


    
  }, [history]);

  return <div ref={chartContainerRef} className="w-full h-full min-h-[250px]" />;
}
