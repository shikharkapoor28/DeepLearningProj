"use client";

import { ReactNode } from "react";
import { motion } from "framer-motion";

export function Panel({ title, children, className = "" }: { title: string, children: ReactNode, className?: string }) {
  return (
    <motion.div 
      suppressHydrationWarning
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={`bg-zinc-900/50 backdrop-blur-xl border border-zinc-800 rounded-xl p-5 flex flex-col gap-4 shadow-2xl relative overflow-hidden ${className}`}
    >
      <div suppressHydrationWarning className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500/20 to-purple-500/20" />
      <h2 suppressHydrationWarning className="text-zinc-300 font-medium tracking-wide text-sm uppercase flex items-center gap-2">
        {title}
      </h2>
      <div suppressHydrationWarning className="flex-1 w-full flex flex-col">
        {children}
      </div>
    </motion.div>
  );
}
