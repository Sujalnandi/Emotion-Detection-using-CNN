import type { ReactNode } from "react";
import Sidebar from "../components/Sidebar";

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="relative flex min-h-screen overflow-hidden bg-ink">
      <div className="pointer-events-none absolute inset-0 bg-grid-radial" />
      <div className="pointer-events-none absolute inset-0 ai-grid opacity-40" />
      <Sidebar />
      <main className="relative z-10 flex-1 p-4 sm:p-6 lg:p-8">{children}</main>
    </div>
  );
}
