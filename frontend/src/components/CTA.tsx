import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Button from "./Button";

export default function CTA() {
  const navigate = useNavigate();

  return (
    <section className="px-4 pb-16 pt-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl">
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true, amount: 0.45 }}
          transition={{ duration: 0.45 }}
          className="rounded-2xl bg-gradient-to-r from-blue-500/45 via-violet-500/40 to-cyan-400/45 p-[1px] shadow-[0_0_60px_rgba(59,130,246,0.24)]"
        >
          <div className="rounded-2xl border border-white/10 bg-slate-950/85 px-6 py-10 text-center backdrop-blur-xl sm:px-10">
            <h2 className="text-3xl font-bold text-white sm:text-4xl">Ready to Start Analyzing?</h2>
            <p className="mx-auto mt-3 max-w-2xl text-sm text-slate-300 sm:text-base">
              Launch detection workflows in seconds and showcase a polished AI product experience from day one.
            </p>
            <div className="mt-7 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Button onClick={() => navigate("/detect")}>Start Detection</Button>
              <Button variant="secondary" onClick={() => navigate("/dashboard")}>View Dashboard</Button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
