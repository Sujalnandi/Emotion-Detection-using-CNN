import { motion } from "framer-motion";
import { ArrowRight, Brain, Camera, Sparkles } from "lucide-react";

const steps = [
  {
    number: "01",
    title: "Upload or Use Webcam",
    description: "Provide an image or connect your camera feed to start emotion scanning in seconds.",
    icon: Camera,
  },
  {
    number: "02",
    title: "AI Analyzes",
    description: "Our model processes facial landmarks and expression patterns with high-speed inference.",
    icon: Brain,
  },
  {
    number: "03",
    title: "Get Insights",
    description: "View emotion probabilities and actionable interpretation in a clear visual format.",
    icon: Sparkles,
  },
];

export default function HowItWorks() {
  return (
    <section className="px-4 py-14 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.35 }}
          transition={{ duration: 0.45 }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold text-white sm:text-4xl">How It Works</h2>
        </motion.div>

        <div className="mt-10 grid gap-4 lg:grid-cols-[1fr_auto_1fr_auto_1fr] lg:items-stretch">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const showArrow = index < steps.length - 1;

            return (
              <div key={step.title} className="contents">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.25 }}
                  transition={{ duration: 0.42, delay: index * 0.08 }}
                  className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl"
                >
                  <span className="pointer-events-none absolute right-3 top-1 text-7xl font-bold text-white/5">
                    {step.number}
                  </span>
                  <div className="relative flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-300/30 bg-cyan-400/10 text-cyan-200">
                    <Icon size={20} />
                  </div>
                  <h3 className="relative mt-4 text-lg font-semibold text-white">{step.title}</h3>
                  <p className="relative mt-2 text-sm leading-relaxed text-slate-300">{step.description}</p>
                </motion.div>

                {showArrow ? (
                  <div className="hidden items-center justify-center text-cyan-200/80 lg:flex">
                    <ArrowRight size={22} />
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
