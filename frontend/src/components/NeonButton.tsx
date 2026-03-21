import { motion } from "framer-motion";
import type { HTMLMotionProps } from "framer-motion";

interface NeonButtonProps extends HTMLMotionProps<"button"> {
  label: string;
}

export default function NeonButton({ label, className = "", ...props }: NeonButtonProps) {
  return (
    <motion.button
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      transition={{ type: "spring", stiffness: 260, damping: 18 }}
      className={[
        "rounded-xl bg-gradient-to-r from-cyan-300 via-blue-500 to-purple-500",
        "px-5 py-2.5 text-sm font-semibold text-slate-950",
        "shadow-neon hover:brightness-110",
        className,
      ].join(" ")}
      {...props}
    >
      {label}
    </motion.button>
  );
}
