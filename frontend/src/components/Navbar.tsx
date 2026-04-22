import { BrainCircuit, Sparkles } from "lucide-react";
import { Link, NavLink } from "react-router-dom";
import Button from "./Button";

const navItems = [
  { label: "Home", to: "/" },
  { label: "Detect", to: "/detect" },
  { label: "Dashboard", to: "/dashboard" },
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-slate-950/70 backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <Link to="/" className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-cyan-300/30 bg-gradient-to-br from-blue-500/30 via-violet-500/20 to-cyan-400/30 text-cyan-200">
            <BrainCircuit size={20} />
          </div>
          <span className="text-lg font-bold tracking-wide text-white">EmotionAI</span>
        </Link>

        <nav className="hidden items-center gap-2 md:flex">
          {navItems.map((item) => {
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  [
                    "rounded-full px-4 py-2 text-sm font-medium transition-colors duration-300",
                    isActive
                      ? "bg-white/10 text-cyan-200 ring-1 ring-cyan-300/30"
                      : "text-slate-300 hover:text-white",
                  ].join(" ")
                }
              >
                {item.label}
              </NavLink>
            );
          })}
        </nav>

        <Link to="/detect" className="hidden sm:inline-flex">
          <Button className="hidden sm:inline-flex" icon={<Sparkles size={16} />}>
            Start Detection
          </Button>
        </Link>
      </div>
    </header>
  );
}
