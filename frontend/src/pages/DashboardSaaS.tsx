import { motion } from "framer-motion";
import Card from "../components/Card";

const metrics = [
  { label: "Predictions Today", value: "2,148" },
  { label: "Avg Confidence", value: "92.4%" },
  { label: "Top Emotion", value: "Happy" },
];

export default function DashboardSaaS() {
  return (
    <section className="px-4 py-12 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="mb-6"
        >
          <h1 className="text-3xl font-bold text-white sm:text-4xl">Dashboard</h1>
          <p className="mt-2 text-sm text-slate-300">Mock analytics snapshot for EmotionAI demos.</p>
        </motion.div>

        <div className="grid gap-4 sm:grid-cols-3">
          {metrics.map((metric) => (
            <Card key={metric.label}>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">{metric.label}</p>
              <p className="mt-3 text-2xl font-bold text-cyan-200">{metric.value}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
