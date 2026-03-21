import GlassPanel from "../components/GlassPanel";

const stats = [
  { emotion: "Angry", count: 3995 },
  { emotion: "Disgust", count: 436 },
  { emotion: "Fear", count: 4097 },
  { emotion: "Happy", count: 7215 },
  { emotion: "Neutral", count: 4965 },
  { emotion: "Sad", count: 4830 },
  { emotion: "Surprise", count: 3171 },
];

const max = Math.max(...stats.map((s) => s.count));

export default function Dataset() {
  return (
    <div className="space-y-8">
      <GlassPanel
        title="Dataset Overview"
        subtitle="Class distribution and sample references for emotion categories."
      >
        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="text-sm font-medium text-slate-200">Images per Emotion</p>
            <div className="mt-4 space-y-3">
              {stats.map((s) => (
                <div key={s.emotion}>
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <span>{s.emotion}</span>
                    <span>{s.count}</span>
                  </div>
                  <div className="mt-1 h-2.5 rounded-full bg-white/10">
                    <div
                      className="h-2.5 rounded-full bg-gradient-to-r from-cyan-300 via-blue-500 to-purple-500"
                      style={{ width: `${(s.count / max) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="text-sm font-medium text-slate-200">Dataset Distribution</p>
            <div className="mt-4 grid grid-cols-2 gap-3">
              {stats.map((s) => (
                <div key={s.emotion} className="rounded-xl border border-white/10 bg-white/5 p-3 text-center text-xs text-slate-300">
                  <div className="mx-auto h-10 w-10 rounded-full bg-gradient-to-br from-cyan-300/40 via-blue-500/35 to-purple-500/45" />
                  <p className="mt-2">{s.emotion}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-white/15 bg-slate-900/45 p-4">
          <p className="text-sm font-medium text-slate-200">Example Images (Placeholder Tiles)</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-7">
            {stats.map((s) => (
              <div key={s.emotion} className="aspect-square rounded-xl border border-white/10 bg-gradient-to-br from-slate-800 to-slate-700 p-2">
                <div className="h-full w-full rounded-lg bg-gradient-to-br from-cyan-300/15 via-blue-500/12 to-purple-500/15" />
                <p className="mt-2 text-center text-[11px] text-slate-300">{s.emotion}</p>
              </div>
            ))}
          </div>
        </div>
      </GlassPanel>
    </div>
  );
}
