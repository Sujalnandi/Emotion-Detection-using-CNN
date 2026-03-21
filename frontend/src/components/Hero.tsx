import { motion } from "framer-motion";
import { Upload } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Button from "./Button";

const emotions = [
  { name: "Happy", color: "bg-yellow-400" },
  { name: "Sad", color: "bg-blue-400" },
  { name: "Angry", color: "bg-rose-400" },
  { name: "Surprise", color: "bg-violet-400" },
  { name: "Fear", color: "bg-orange-400" },
  { name: "Neutral", color: "bg-slate-400" },
];

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0 },
};

export default function Hero() {
  const navigate = useNavigate();

  return (
    <section className="relative overflow-hidden px-4 pb-14 pt-14 sm:px-6 lg:px-8 lg:pt-20">
      <div className="mx-auto max-w-6xl text-center">
        <motion.span
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          transition={{ duration: 0.45 }}
          className="inline-flex items-center rounded-full border border-cyan-300/30 bg-cyan-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200"
        >
          AI-Powered Emotion Analysis
        </motion.span>

        <motion.h1
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mx-auto mt-6 max-w-4xl text-balance text-4xl font-bold leading-tight text-white sm:text-5xl lg:text-6xl"
        >
          AI Facial <span className="bg-gradient-to-r from-blue-400 via-violet-400 to-cyan-300 bg-clip-text text-transparent">Emotion Detection</span>
        </motion.h1>

        <motion.p
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.18 }}
          className="mx-auto mt-5 max-w-2xl text-sm leading-relaxed text-slate-300 sm:text-base"
        >
          Analyze human emotions in real-time using advanced artificial intelligence. Detect subtle expressions from webcam streams or uploaded images with clarity and speed.
        </motion.p>

        <motion.div
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.26 }}
          className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row"
        >
          <Button onClick={() => navigate("/detect")}>Start Detection</Button>
          <Button variant="secondary" icon={<Upload size={16} />} onClick={() => navigate("/detect")}>
            Upload Image
          </Button>
        </motion.div>

        <motion.div
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.34 }}
          className="mt-9 flex flex-wrap items-center justify-center gap-3"
        >
          {emotions.map((emotion) => (
            <div
              key={emotion.name}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-200"
            >
              <span className={`h-2.5 w-2.5 rounded-full ${emotion.color}`} />
              {emotion.name}
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
