import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
}

export default function FeatureCard({ title, description, icon: Icon }: FeatureCardProps) {
  return (
    <motion.article
      whileHover={{ y: -10 }}
      transition={{ duration: 0.25 }}
      className="glass group relative overflow-hidden rounded-3xl p-6 shadow-glass"
    >
      <div className="pointer-events-none absolute -right-12 -top-12 h-36 w-36 rounded-full bg-gradient-to-br from-purple-500/30 via-blue-500/20 to-cyan-400/25 blur-2xl" />
      <div className="relative z-10">
        <div className="inline-flex rounded-xl border border-white/20 bg-white/10 p-3">
          <Icon className="h-6 w-6 text-cyan-200 transition-colors group-hover:text-purple-300" />
        </div>
        <h3 className="mt-4 text-lg font-semibold text-white">{title}</h3>
        <p className="mt-2 text-sm leading-relaxed text-slate-300">{description}</p>
      </div>
    </motion.article>
  );
}
