import { BrainCircuit } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-white/10 px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col items-start justify-between gap-2 sm:flex-row sm:items-center">
        <div className="flex items-center gap-2 text-slate-200">
          <BrainCircuit size={18} className="text-cyan-200" />
          <span className="text-sm font-semibold">EmotionAI</span>
        </div>
        <p className="text-xs text-slate-400">Built for hackathons and demo purposes</p>
      </div>
    </footer>
  );
}
