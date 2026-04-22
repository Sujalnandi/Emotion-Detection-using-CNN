import { motion } from "framer-motion";
import { BarChart3, ScanFace, ShieldCheck, Zap } from "lucide-react";
import Card from "./Card";

const featureItems = [
  {
    title: "Real-Time Detection",
    description: "Analyze live facial expressions instantly through webcam input with responsive AI feedback.",
    icon: ScanFace,
  },
  {
    title: "Analytics Dashboard",
    description: "Track emotion trends over sessions and review visual summaries for better insight.",
    icon: BarChart3,
  },
  {
    title: "Lightning Fast",
    description: "Optimized inference pipeline delivers smooth performance for demos and production UI.",
    icon: Zap,
  },
  {
    title: "Privacy First",
    description: "Client-facing workflows are designed for secure handling and minimal retention of user data.",
    icon: ShieldCheck,
  },
];

export default function Features() {
  return (
    <section className="px-4 py-14 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.45 }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold text-white sm:text-4xl">Powerful Features</h2>
          <p className="mx-auto mt-3 max-w-2xl text-sm text-slate-300 sm:text-base">
            Everything you need to analyze and understand facial emotion signals with a modern AI toolkit.
          </p>
        </motion.div>

        <div className="mt-9 grid gap-4 sm:grid-cols-2 lg:gap-6">
          {featureItems.map((feature, index) => {
            const Icon = feature.icon;

            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.2 }}
                transition={{ duration: 0.42, delay: index * 0.08 }}
              >
                <Card className="h-full">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-300/35 bg-gradient-to-br from-blue-500/25 via-violet-500/20 to-cyan-400/25 text-cyan-200">
                    <Icon size={20} />
                  </div>
                  <h3 className="mt-5 text-lg font-semibold text-white">{feature.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-slate-300">{feature.description}</p>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
