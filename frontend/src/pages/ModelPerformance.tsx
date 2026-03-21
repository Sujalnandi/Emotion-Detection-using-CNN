import GlassPanel from "../components/GlassPanel";

const accTrain = [18, 26, 35, 44, 51, 56];
const lossTrain = [1.9, 1.62, 1.45, 1.31, 1.2, 1.13];

function linePoints(values: number[], maxHeight = 100): string {
  const step = 340 / (values.length - 1);
  const max = Math.max(...values);
  const min = Math.min(...values);
  return values
    .map((v, i) => {
      const normalized = max === min ? 0.5 : (v - min) / (max - min);
      const x = 10 + i * step;
      const y = 10 + (1 - normalized) * maxHeight;
      return `${x},${y}`;
    })
    .join(" ");
}

export default function ModelPerformance() {
  return (
    <div className="space-y-8">
      <GlassPanel
        title="Model Performance"
        subtitle="Training insights, confusion trends, and architecture comparison."
      >
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="text-sm font-medium text-slate-200">Training Accuracy Graph</p>
            <svg viewBox="0 0 360 130" className="mt-4 h-44 w-full">
              <polyline
                fill="none"
                stroke="#22d3ee"
                strokeWidth="3"
                points={linePoints(accTrain)}
              />
            </svg>
          </div>

          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="text-sm font-medium text-slate-200">Loss Graph</p>
            <svg viewBox="0 0 360 130" className="mt-4 h-44 w-full">
              <polyline
                fill="none"
                stroke="#a855f7"
                strokeWidth="3"
                points={linePoints(lossTrain)}
              />
            </svg>
          </div>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="text-sm font-medium text-slate-200">Confusion Matrix</p>
            <div className="mt-4 grid grid-cols-7 gap-2 text-center text-xs text-slate-300">
              {Array.from({ length: 49 }).map((_, idx) => (
                <div
                  key={idx}
                  className="rounded-md border border-white/15 bg-white/5 py-2"
                  style={{ opacity: 0.3 + (idx % 7) * 0.08 }}
                >
                  {Math.floor(30 + ((idx * 7) % 61))}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="text-sm font-medium text-slate-200">Model Comparison</p>
            <table className="mt-3 w-full text-left text-sm text-slate-300">
              <thead>
                <tr className="text-slate-400">
                  <th className="py-2">Model</th>
                  <th className="py-2">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-white/10">
                  <td className="py-2">CNN</td>
                  <td className="py-2 text-cyan-200">56%</td>
                </tr>
                <tr className="border-t border-white/10">
                  <td className="py-2">ResNet50</td>
                  <td className="py-2 text-purple-200">26%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </GlassPanel>
    </div>
  );
}
