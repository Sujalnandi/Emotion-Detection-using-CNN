import type { ReactNode } from "react";

interface GlassPanelProps {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

export default function GlassPanel({ title, subtitle, children, className = "" }: GlassPanelProps) {
  return (
    <section className={`glass neon-border rounded-3xl p-6 md:p-7 shadow-glass ${className}`}>
      {(title || subtitle) && (
        <header className="mb-5">
          {title && <h2 className="text-2xl font-semibold text-slate-50">{title}</h2>}
          {subtitle && <p className="mt-1 text-sm text-slate-300">{subtitle}</p>}
        </header>
      )}
      {children}
    </section>
  );
}
