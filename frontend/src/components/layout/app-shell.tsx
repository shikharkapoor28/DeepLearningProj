export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans selection:bg-blue-500/30">
      <header className="h-14 border-b border-zinc-900 bg-zinc-950/50 backdrop-blur-md sticky top-0 z-50 flex items-center px-6">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse shadow-[0_0_10px_rgba(59,130,246,0.6)]" />
          <h1 className="font-bold text-sm tracking-widest text-zinc-200 uppercase">QuantRL <span className="text-zinc-500 font-medium">| Trading Engine</span></h1>
        </div>
      </header>
      <main className="p-6 max-w-[1600px] mx-auto">
        {children}
      </main>
    </div>
  );
}
