import { motion } from "framer-motion";
import FeatureCard from "../components/FeatureCard";
import GlassPanel from "../components/GlassPanel";
import { icons } from "../utils/icons";

export default function Dashboard() {
  return (
    <div className="space-y-8">
      <GlassPanel className="relative overflow-hidden">
        <div className="pointer-events-none absolute -right-16 -top-20 h-60 w-60 rounded-full bg-gradient-to-br from-purple-500/30 via-blue-500/25 to-cyan-400/25 blur-3xl" />
        <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/70">Facial Intelligence Suite</p>
          <h1 className="mt-4 max-w-3xl text-4xl font-semibold text-white md:text-5xl">
            Facial Emotion Detection AI
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-slate-300 md:text-base">
            A deep learning system capable of recognizing human emotions from facial expressions using convolutional neural networks and computer vision.
          </p>
        </motion.div>
      </GlassPanel>

      <section className="grid gap-5 md:grid-cols-3">
        <FeatureCard
          title="Image Emotion Detection"
          description="Upload a face image and inspect prediction confidence across all seven emotion classes."
          icon={icons.Image}
        />
        <FeatureCard
          title="Real-Time Emotion Detection"
          description="Stream webcam frames and view live facial emotion overlays with confidence scores."
          icon={icons.Camera}
        />
        <FeatureCard
          title="Model Analytics"
          description="Track training behavior, confusion matrix quality, and model-level comparison metrics."
          icon={icons.BarChart}
        />
      </section>
    </div>
  );
}
