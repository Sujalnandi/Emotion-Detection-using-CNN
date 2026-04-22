import { motion } from "framer-motion";
import { cn } from "../utils/cn";
import type { HTMLMotionProps } from "framer-motion";
import type { ReactNode } from "react";

type ButtonVariant = "primary" | "secondary";

interface ButtonProps extends HTMLMotionProps<"button"> {
  children: ReactNode;
  variant?: ButtonVariant;
  icon?: ReactNode;
}

export default function Button({ children, variant = "primary", icon, className, ...props }: ButtonProps) {
  const baseStyles =
    "inline-flex items-center justify-center gap-2 rounded-2xl px-5 py-3 text-sm font-semibold transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/70 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950";

  const variantStyles =
    variant === "primary"
      ? "bg-gradient-to-r from-blue-500 via-violet-500 to-cyan-400 text-slate-950 shadow-[0_10px_30px_rgba(56,189,248,0.32)] hover:scale-[1.02] hover:brightness-110"
      : "border border-white/20 bg-white/5 text-slate-100 backdrop-blur-md hover:border-cyan-300/40 hover:bg-white/10";

  return (
    <motion.button
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 300, damping: 18 }}
      className={cn(baseStyles, variantStyles, className)}
      {...props}
    >
      {icon}
      {children}
    </motion.button>
  );
}
