import { BrowserRouter, Route, Routes } from "react-router-dom";
import CTA from "./components/CTA";
import Features from "./components/Features";
import Footer from "./components/Footer";
import Hero from "./components/Hero";
import HowItWorks from "./components/HowItWorks";
import Navbar from "./components/Navbar";
import DashboardSaaS from "./pages/DashboardSaaS";
import Detect from "./pages/Detect";

function Home() {
  return (
    <main>
      <Hero />
      <Features />
      <HowItWorks />
      <CTA />
    </main>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -left-24 top-20 h-72 w-72 rounded-full bg-blue-500/25 blur-3xl" />
          <div className="absolute -right-20 top-40 h-72 w-72 rounded-full bg-violet-500/20 blur-3xl" />
          <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-cyan-500/20 blur-3xl" />
        </div>

        <div className="relative z-10">
          <Navbar />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/detect" element={<Detect />} />
            <Route path="/dashboard" element={<DashboardSaaS />} />
          </Routes>
          <Footer />
        </div>

        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.06),transparent_45%)]" />
      </div>
    </BrowserRouter>
  );
}
