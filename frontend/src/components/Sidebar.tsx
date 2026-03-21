import { motion } from "framer-motion";
import { NavLink } from "react-router-dom";
import { useMemo, useState } from "react";
import { icons } from "../utils/icons";

const navItems = [
  { to: "/", label: "Dashboard", icon: icons.Home },
  { to: "/image", label: "Image Emotion Detection", icon: icons.Image },
  { to: "/camera", label: "Real-Time Camera Detection", icon: icons.Camera },
  { to: "/performance", label: "Model Performance", icon: icons.BarChart },
  { to: "/dataset", label: "Dataset Overview", icon: icons.Dataset },
  { to: "/about", label: "About Project", icon: icons.About },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  const widthClass = useMemo(() => (collapsed ? "w-20" : "w-72"), [collapsed]);

  return (
    <motion.aside
      animate={{ width: collapsed ? 80 : 288 }}
      transition={{ duration: 0.25 }}
      className={`${widthClass} hidden h-screen shrink-0 border-r border-white/10 bg-slate-950/70 backdrop-blur-xl lg:flex lg:flex-col`}
    >
      <div className="flex items-center justify-between px-4 py-5">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-gradient-to-br from-cyan-300 via-blue-500 to-purple-500 p-2">
            <icons.Sparkles className="h-5 w-5 text-slate-950" />
          </div>
          {!collapsed && (
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-cyan-200/75">AI LAB</p>
              <p className="text-sm font-semibold text-white">Emotion Interface</p>
            </div>
          )}
        </div>

        <button
          className="rounded-full border border-white/15 bg-white/10 p-2 text-slate-200 hover:bg-white/20"
          onClick={() => setCollapsed((prev) => !prev)}
          aria-label="Toggle sidebar"
        >
          {collapsed ? <icons.ChevronRight className="h-4 w-4" /> : <icons.ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      <nav className="flex-1 space-y-2 px-3 py-3">
        {navItems.map((item) => (
          <NavLink
            key={item.label}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              [
                "group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all",
                isActive
                  ? "bg-gradient-to-r from-cyan-400/25 via-blue-500/15 to-purple-500/25 text-white shadow-neon"
                  : "text-slate-300 hover:bg-white/10 hover:text-white",
              ].join(" ")
            }
          >
            <item.icon className="h-5 w-5 shrink-0 text-cyan-200" />
            {!collapsed && <span className="truncate">{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="m-3 rounded-2xl border border-white/15 bg-white/5 p-4">
        {!collapsed ? (
          <>
            <p className="text-xs text-cyan-200">Inference Status</p>
            <p className="mt-2 text-sm text-slate-300">Camera stream and API endpoint connected.</p>
          </>
        ) : (
          <icons.Activity className="mx-auto h-5 w-5 text-cyan-200" />
        )}
      </div>
    </motion.aside>
  );
}
