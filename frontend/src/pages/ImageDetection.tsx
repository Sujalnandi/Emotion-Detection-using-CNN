import { useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import GlassPanel from "../components/GlassPanel";
import NeonButton from "../components/NeonButton";
import type { FaceBox, FacePrediction, PredictionResponse } from "../utils/api";
import { predictImage } from "../utils/api";

function asPercentBox(box: FaceBox): FaceBox {
  if (box.x <= 1 && box.y <= 1 && box.w <= 1 && box.h <= 1) {
    return { x: box.x * 100, y: box.y * 100, w: box.w * 100, h: box.h * 100 };
  }
  return box;
}

export default function ImageDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const facePredictions: FacePrediction[] = useMemo(
    () => result?.faces_predictions ?? [],
    [result]
  );

  const onFile = (next: File) => {
    setFile(next);
    setResult(null);
    setPreview(URL.createObjectURL(next));
  };

  const onAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    const prediction = await predictImage(file);
    setResult(prediction);
    setLoading(false);
  };

  return (
    <div className="space-y-8">
      <GlassPanel
        title="Image Emotion Detection"
        subtitle="Upload an image and get emotion probabilities with face bounding boxes."
      >
        <div
          className="rounded-2xl border-2 border-dashed border-cyan-300/35 bg-white/5 p-6 text-center"
          onDrop={(event) => {
            event.preventDefault();
            const dropped = event.dataTransfer.files?.[0];
            if (dropped) onFile(dropped);
          }}
          onDragOver={(event) => event.preventDefault()}
        >
          <p className="text-sm text-slate-300">Drag and drop image here or click to upload.</p>
          <div className="mt-4">
            <NeonButton label="Choose Image" onClick={() => inputRef.current?.click()} />
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              accept="image/*"
              onChange={(event) => {
                const next = event.target.files?.[0];
                if (next) onFile(next);
              }}
            />
            <NeonButton
              label={loading ? "Analyzing..." : "Analyze Emotion"}
              disabled={!file || loading}
              onClick={onAnalyze}
              className="ml-3 disabled:cursor-not-allowed disabled:opacity-45"
            />
          </div>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-[1.3fr_1fr]">
          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="mb-3 text-xs uppercase tracking-[0.3em] text-cyan-200/70">Preview + Bounding Boxes</p>
            <div className="relative min-h-[280px] overflow-hidden rounded-xl border border-white/10 bg-slate-950/50">
              {preview ? (
                <>
                  <img src={preview} alt="Uploaded face" className="h-full w-full object-contain" />
                  {facePredictions.map((face) => {
                    const b = asPercentBox(face.bbox);
                    const style =
                      b.x <= 100
                        ? {
                            left: `${b.x}%`,
                            top: `${b.y}%`,
                            width: `${b.w}%`,
                            height: `${b.h}%`,
                          }
                        : { left: b.x, top: b.y, width: b.w, height: b.h };
                    return (
                      <div key={face.id} className="absolute" style={style}>
                        <div className="h-full w-full rounded-md border-2 border-cyan-300 shadow-neon" />
                        <div className="absolute -top-6 left-0 rounded-md bg-slate-950/80 px-2 py-0.5 text-[11px] text-cyan-200">
                          Face {face.id} - {face.emotion} ({(face.confidence * 100).toFixed(0)}%)
                        </div>
                      </div>
                    );
                  })}
                </>
              ) : (
                <div className="flex h-[280px] items-center justify-center text-sm text-slate-500">No image selected</div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/70">Prediction Result</p>
            {result ? (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-4 space-y-4">
                {facePredictions.length === 0 && (
                  <p className="text-sm text-slate-400">No faces detected in this image.</p>
                )}

                {facePredictions.map((face) => {
                  const probs = Object.entries(face.probabilityMap).sort((a, b) => b[1] - a[1]);
                  return (
                    <div key={face.id} className="rounded-xl border border-white/10 bg-white/5 p-4">
                      <p className="text-xs text-slate-400">Face {face.id}</p>
                      <p className="mt-1 text-lg font-semibold text-white">Emotion: {face.emotion}</p>
                      <p className="text-sm text-cyan-200">Confidence: {(face.confidence * 100).toFixed(1)}%</p>

                      <div className="mt-3 space-y-2">
                        {probs.map(([emotion, value]) => (
                          <div key={`${face.id}-${emotion}`}>
                            <div className="flex items-center justify-between text-xs text-slate-300">
                              <span>{emotion}</span>
                              <span>{(value * 100).toFixed(1)}%</span>
                            </div>
                            <div className="mt-1 h-2.5 rounded-full bg-white/10">
                              <div
                                className="h-2.5 rounded-full bg-gradient-to-r from-cyan-300 via-blue-500 to-purple-500"
                                style={{ width: `${Math.max(value * 100, 2)}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </motion.div>
            ) : (
              <p className="mt-4 text-sm text-slate-400">Run analysis to view emotion probabilities.</p>
            )}
          </div>
        </div>
      </GlassPanel>
    </div>
  );
}
