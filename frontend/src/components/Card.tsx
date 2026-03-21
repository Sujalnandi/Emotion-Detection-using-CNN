import { motion } from "framer-motion";
import { cn } from "../utils/cn";
import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
}

export default function Card({ children, className }: CardProps) {
  return (
    <motion.div
      whileHover={{ y: -6 }}
      transition={{ type: "spring", stiffness: 260, damping: 20 }}
      className={cn(
        "rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl shadow-[0_24px_64px_rgba(2,6,23,0.45)] transition-all duration-300",
        className
      )}
    >
      {children}
    </motion.div>
  );
}
