import GlassPanel from "../components/GlassPanel";

export default function About() {
  return (
    <div className="space-y-8">
      <GlassPanel title="About Project" subtitle="Facial Emotion Detection System - B.Tech AI/ML Final-Year Project">
        <div className="grid gap-6 md:grid-cols-2">
          <article className="rounded-2xl border border-white/15 bg-slate-900/45 p-4 text-sm text-slate-300">
            <h3 className="mb-2 text-base font-semibold text-white">Project Description</h3>
            <p>
              This system detects human emotions from facial images and real-time webcam streams using deep learning and computer vision. It integrates OpenCV face detection with CNN-based emotion classification.
            </p>
          </article>

          <article className="rounded-2xl border border-white/15 bg-slate-900/45 p-4 text-sm text-slate-300">
            <h3 className="mb-2 text-base font-semibold text-white">Technology Stack</h3>
            <ul className="list-disc space-y-1 pl-5">
              <li>Python, TensorFlow, Keras, OpenCV</li>
              <li>React, TypeScript, Tailwind CSS</li>
              <li>Framer Motion, Lucide Icons, React Router</li>
            </ul>
          </article>

          <article className="rounded-2xl border border-white/15 bg-slate-900/45 p-4 text-sm text-slate-300">
            <h3 className="mb-2 text-base font-semibold text-white">Model Architecture</h3>
            <p>
              Convolutional neural network with multiple Conv2D, pooling, dropout, and dense layers for 7-class softmax prediction. Optional ResNet50 transfer learning variant is used for baseline comparison.
            </p>
          </article>

          <article className="rounded-2xl border border-white/15 bg-slate-900/45 p-4 text-sm text-slate-300">
            <h3 className="mb-2 text-base font-semibold text-white">Author Information</h3>
            <p>Project: Facial Emotion Detection System</p>
            <p>Program: B.Tech AI/ML</p>
            <p>Role: Final-year Capstone Implementation</p>
          </article>
        </div>
      </GlassPanel>
    </div>
  );
}
