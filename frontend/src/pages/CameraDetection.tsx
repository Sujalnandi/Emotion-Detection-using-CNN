import { useEffect, useMemo, useRef, useState } from "react";
import GlassPanel from "../components/GlassPanel";
import NeonButton from "../components/NeonButton";
import { predictFrame, type PredictionResponse } from "../utils/api";

export default function CameraDetection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timerRef = useRef<number | null>(null);

  const [streaming, setStreaming] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const facePredictions = useMemo(() => result?.faces_predictions ?? [], [result]);

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      setStreaming(true);
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((track) => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setStreaming(false);
    setDetecting(false);
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const detectOnce = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) return;
    const prediction = await predictFrame(blob);
    setResult(prediction);
  };

  const toggleRealtime = () => {
    if (detecting) {
      setDetecting(false);
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    setDetecting(true);
    detectOnce();
    timerRef.current = window.setInterval(detectOnce, 1300);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="space-y-8">
      <GlassPanel
        title="Real-Time Camera Detection"
        subtitle="Stream webcam frames and display detected emotion with confidence and face boxes."
      >
        <div className="mb-5 flex flex-wrap gap-3">
          <NeonButton label={streaming ? "Restart Camera" : "Start Camera"} onClick={startCamera} />
          <NeonButton
            label={detecting ? "Stop Real-Time" : "Start Real-Time"}
            disabled={!streaming}
            onClick={toggleRealtime}
            className="disabled:opacity-40"
          />
          <NeonButton label="Stop Camera" onClick={stopCamera} className="bg-gradient-to-r from-slate-300 to-slate-100" />
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
          <div className="relative overflow-hidden rounded-2xl border border-white/15 bg-slate-950/55">
            <video ref={videoRef} autoPlay playsInline className="h-[360px] w-full object-cover" />
            <canvas ref={canvasRef} className="hidden" />

            {facePredictions.map((face) => (
              <div
                key={face.id}
                className="absolute"
                style={{
                  left: `${Math.max(0, face.bbox.x <= 1 ? face.bbox.x * 100 : face.bbox.x)}%`,
                  top: `${Math.max(0, face.bbox.y <= 1 ? face.bbox.y * 100 : face.bbox.y)}%`,
                  width: `${face.bbox.w <= 1 ? face.bbox.w * 100 : face.bbox.w}%`,
                  height: `${face.bbox.h <= 1 ? face.bbox.h * 100 : face.bbox.h}%`,
                }}
              >
                <div className="h-full w-full border-2 border-cyan-300 shadow-neon" />
                <div className="absolute -top-6 left-0 rounded-md bg-slate-950/80 px-2 py-0.5 text-[11px] text-cyan-200">
                  Face {face.id} - {face.emotion} ({(face.confidence * 100).toFixed(0)}%)
                </div>
              </div>
            ))}

            {result && (
              <div className="absolute left-3 top-3 rounded-lg border border-white/20 bg-slate-950/70 px-3 py-1 text-sm text-cyan-200">
                Detected Emotion: {result.emotion} ({(result.confidence * 100).toFixed(1)}%)
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-4">
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/70">Live Face Predictions</p>
            <div className="mt-4 space-y-3">
              {facePredictions.length === 0 && <p className="text-sm text-slate-400">Start detection to view live scores.</p>}
              {facePredictions.map((face) => {
                const probabilities = Object.entries(face.probabilityMap).sort((a, b) => b[1] - a[1]);
                return (
                  <div key={face.id} className="rounded-xl border border-white/10 bg-white/5 p-3">
                    <p className="text-sm font-semibold text-white">
                      Face {face.id}: {face.emotion} ({(face.confidence * 100).toFixed(1)}%)
                    </p>
                    <div className="mt-2 space-y-2">
                      {probabilities.map(([emotion, score]) => (
                        <div key={`${face.id}-${emotion}`}>
                          <div className="flex items-center justify-between text-xs text-slate-300">
                            <span>{emotion}</span>
                            <span>{(score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="mt-1 h-2.5 rounded-full bg-white/10">
                            <div
                              className="h-2.5 rounded-full bg-gradient-to-r from-cyan-300 via-blue-500 to-purple-500"
                              style={{ width: `${Math.max(score * 100, 2)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </GlassPanel>
    </div>
  );
}
