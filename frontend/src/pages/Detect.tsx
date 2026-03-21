import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { motion } from "framer-motion";
import { Camera, CameraOff, ImageUp, Loader2, Sparkles, TriangleAlert } from "lucide-react";
import Button from "../components/Button";
import Card from "../components/Card";

type PredictEmotionResponse = {
  emotion: string;
  confidence: number;
  confidencePercent: number;
  allProbabilities: Record<string, number>;
  probabilities: Record<string, number>;
  faces: Array<{
    id: number;
    emotion: string;
    confidence: number;
    bbox: { x: number; y: number; w: number; h: number };
  }>;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

function normalizeProbabilities(raw: unknown): Record<string, number> {
  if (!raw || typeof raw !== "object") return {};
  return Object.fromEntries(
    Object.entries(raw as Record<string, unknown>).map(([key, value]) => {
      const num = typeof value === "number" && !Number.isNaN(value) ? value : 0;
      return [key.toLowerCase(), num > 1 ? Math.min(100, num) / 100 : Math.max(0, num)];
    })
  );
}

function toUiResult(payload: Record<string, unknown>): PredictEmotionResponse {
  const rawConfidence = typeof payload.confidence === "number" ? payload.confidence : 0;
  const confidencePercent = rawConfidence > 1 ? rawConfidence : rawConfidence * 100;
  const probabilities = normalizeProbabilities(payload.probabilities ?? payload.all_probabilities);

  const faces = Array.isArray(payload.faces)
    ? payload.faces.map((face, index) => {
        const row = (face ?? {}) as Record<string, unknown>;
        const bbox = (row.bbox ?? {}) as Record<string, unknown>;
        return {
          id: Number(row.id ?? index + 1),
          emotion: String(row.emotion ?? "Unknown"),
          confidence:
            typeof row.confidence === "number"
              ? (row.confidence > 1 ? Math.min(100, row.confidence) / 100 : Math.max(0, row.confidence))
              : 0,
          bbox: {
            x: Number(bbox.x ?? 0),
            y: Number(bbox.y ?? 0),
            w: Number(bbox.w ?? 0),
            h: Number(bbox.h ?? 0),
          },
        };
      })
    : [];

  return {
    emotion: String(payload.emotion ?? "Unknown"),
    confidence: confidencePercent / 100,
    confidencePercent,
    allProbabilities: probabilities,
    probabilities,
    faces,
  };
}

export default function Detect() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<number | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictEmotionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [realtimeActive, setRealtimeActive] = useState(false);
  const [realtimeLoading, setRealtimeLoading] = useState(false);
  const [realtimeError, setRealtimeError] = useState<string | null>(null);
  const [realtimeResult, setRealtimeResult] = useState<PredictEmotionResponse | null>(null);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please choose an image before starting detection.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = (await response.json()) as Record<string, unknown>;

      if (!response.ok) {
        console.error(data);
        alert("API Error");
        setError("Prediction request failed.");
        setResult(null);
        return;
      }

      if (typeof data.error === "string") {
        setError(data.error);
        setResult(null);
        return;
      }

      setResult(toUiResult(data));
    } catch (submissionError) {
      const message = submissionError instanceof Error ? submissionError.message : "Prediction request failed.";
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setStreaming(true);
      setRealtimeError(null);
    } catch {
      setRealtimeError("Unable to access webcam. Check browser camera permissions.");
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((track) => track.stop());
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setStreaming(false);
    setRealtimeActive(false);
    setRealtimeLoading(false);
  };

  const captureAndPredict = async () => {
    if (!videoRef.current || !canvasRef.current || realtimeLoading) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    canvas.width = 640;
    canvas.height = Math.round((video.videoHeight / video.videoWidth) * 640);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frameBase64 = canvas.toDataURL("image/jpeg", 0.72);

    setRealtimeLoading(true);
    try {
      const response = await fetch(`${API_BASE}/realtime`, {
        method: "POST",
        body: JSON.stringify({ image: frameBase64 }),
        headers: { "Content-Type": "application/json" },
      });

      const data = (await response.json()) as Record<string, unknown>;

      if (!response.ok) {
        console.error(data);
        alert("API Error");
        setRealtimeError("Realtime prediction failed.");
        return;
      }

      if (typeof data.error === "string") {
        setRealtimeError(data.error);
        setRealtimeResult(null);
        return;
      }

      setRealtimeResult(toUiResult(data));
      setRealtimeError(null);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Realtime prediction failed.";
      setRealtimeError(message);
    } finally {
      setRealtimeLoading(false);
    }
  };

  const toggleRealtime = () => {
    if (!streaming) return;

    if (realtimeActive) {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setRealtimeActive(false);
      return;
    }

    setRealtimeActive(true);
    captureAndPredict();
    intervalRef.current = window.setInterval(captureAndPredict, 1000);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
      }
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const probabilityEntries = Object.entries(result?.allProbabilities ?? {}).sort((a, b) => b[1] - a[1]);
  const realtimeProbabilityEntries = Object.entries(realtimeResult?.allProbabilities ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <section className="px-4 py-12 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="text-center"
        >
          <h1 className="text-3xl font-bold text-white sm:text-4xl">Detect Emotion</h1>
          <p className="mx-auto mt-2 max-w-2xl text-sm text-slate-300 sm:text-base">
            Upload a face image and send it to the AI API endpoint for live emotion and confidence results.
          </p>
        </motion.div>

        <Card>
          <form className="space-y-5" onSubmit={handleSubmit}>
            <label
              htmlFor="emotion-image"
              className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-cyan-300/40 bg-white/5 px-6 py-10 text-center transition hover:border-cyan-200/60 hover:bg-white/10"
            >
              <ImageUp className="mb-3 text-cyan-200" size={24} />
              <span className="text-sm font-medium text-slate-200">Choose image file</span>
              <span className="mt-1 text-xs text-slate-400">JPG, PNG, WEBP supported</span>
              <input id="emotion-image" type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
            </label>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="truncate text-sm text-slate-300">
                {selectedFile ? `Selected: ${selectedFile.name}` : "No file selected"}
              </p>
              <Button type="submit" disabled={!selectedFile || loading} icon={loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}>
                {loading ? "Analyzing..." : "Start Detection"}
              </Button>
            </div>
          </form>
        </Card>

        <div className="grid gap-4 md:grid-cols-2">
          <Card className="min-h-[220px]">
            <h2 className="text-sm font-semibold uppercase tracking-[0.22em] text-cyan-200/90">Image Preview</h2>
            <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-slate-900/60">
              {previewUrl ? (
                <img src={previewUrl} alt="Selected file preview" className="h-64 w-full object-contain" />
              ) : (
                <div className="flex h-64 items-center justify-center text-sm text-slate-500">Upload an image to preview</div>
              )}
            </div>
          </Card>

          <Card className="min-h-[220px]">
            <h2 className="text-sm font-semibold uppercase tracking-[0.22em] text-cyan-200/90">Prediction Result</h2>

            {loading ? (
              <div className="mt-5 flex items-center gap-3 text-sm text-slate-300">
                <Loader2 size={18} className="animate-spin text-cyan-200" />
                Waiting for /predict response...
              </div>
            ) : null}

            {error ? (
              <div className="mt-5 rounded-xl border border-rose-300/25 bg-rose-400/10 p-4 text-sm text-rose-100">
                <div className="flex items-start gap-2">
                  <TriangleAlert size={16} className="mt-0.5 text-rose-300" />
                  <p>{error}</p>
                </div>
              </div>
            ) : null}

            {result ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-5 space-y-4"
              >
                <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Emotion</p>
                  <p className="mt-2 text-2xl font-bold text-white">{result.emotion}</p>
                </div>

                <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Confidence</p>
                  <p className="mt-2 text-2xl font-bold text-cyan-200">{result.confidencePercent.toFixed(2)}%</p>
                  <div className="mt-3 h-2.5 rounded-full bg-white/10">
                    <div
                      className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 via-violet-500 to-cyan-400"
                      style={{ width: `${Math.max(4, Math.min(100, result.confidencePercent))}%` }}
                    />
                  </div>
                </div>

                <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">All Probabilities</p>
                  <div className="mt-3 space-y-2">
                    {probabilityEntries.length === 0 ? (
                      <p className="text-sm text-slate-400">No class probability map returned.</p>
                    ) : (
                      probabilityEntries.map(([emotion, score]) => (
                        <div key={emotion}>
                          <div className="flex items-center justify-between text-xs text-slate-300">
                            <span className="capitalize">{emotion}</span>
                            <span>{(score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="mt-1 h-2 rounded-full bg-white/10">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-blue-500 via-violet-500 to-cyan-400"
                              style={{ width: `${Math.max(2, Math.min(100, score * 100))}%` }}
                            />
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </motion.div>
            ) : null}

            {!loading && !error && !result ? (
              <p className="mt-5 text-sm text-slate-400">Submit an image to see predicted emotion and confidence.</p>
            ) : null}
          </Card>
        </div>

        <Card>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-[0.22em] text-cyan-200/90">Real-Time Webcam Detection</h2>
              <p className="mt-1 text-sm text-slate-300">Capture frame every 1 second and send base64 payload to /realtime.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button icon={<Camera size={16} />} onClick={startCamera}>
                {streaming ? "Restart Camera" : "Start Camera"}
              </Button>
              <Button variant="secondary" onClick={toggleRealtime} disabled={!streaming}>
                {realtimeActive ? "Stop Detection" : "Start Detection"}
              </Button>
              <Button variant="secondary" icon={<CameraOff size={16} />} onClick={stopCamera}>
                Stop Camera
              </Button>
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1.35fr_1fr]">
            <div className="relative overflow-hidden rounded-xl border border-white/10 bg-slate-900/70">
              <video ref={videoRef} autoPlay playsInline muted className="h-[340px] w-full object-cover" />
              <canvas ref={canvasRef} className="hidden" />

              {(realtimeResult?.faces ?? []).map((face) => {
                const x = Math.max(0, face.bbox.x * 100);
                const y = Math.max(0, face.bbox.y * 100);
                const w = Math.max(0, face.bbox.w * 100);
                const h = Math.max(0, face.bbox.h * 100);
                return (
                  <div
                    key={`face-${face.id}`}
                    className="absolute"
                    style={{ left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%` }}
                  >
                    <div className="h-full w-full rounded-md border-2 border-cyan-300 shadow-[0_0_20px_rgba(34,211,238,0.35)]" />
                    <div className="absolute -top-6 left-0 rounded bg-slate-950/80 px-2 py-0.5 text-[11px] text-cyan-200">
                      {face.emotion} ({(face.confidence * 100).toFixed(0)}%)
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Live Prediction</p>

              {realtimeLoading ? (
                <div className="mt-3 flex items-center gap-2 text-sm text-slate-300">
                  <Loader2 size={15} className="animate-spin text-cyan-200" />
                  Processing current frame...
                </div>
              ) : null}

              {realtimeError ? (
                <div className="mt-3 rounded-lg border border-rose-300/25 bg-rose-400/10 p-3 text-sm text-rose-100">
                  {realtimeError}
                </div>
              ) : null}

              {realtimeResult ? (
                <div className="mt-3 space-y-3">
                  <div className="rounded-lg border border-white/10 bg-slate-950/45 p-3">
                    <p className="text-xs text-slate-400">Emotion</p>
                    <p className="mt-1 text-xl font-semibold text-white">{realtimeResult.emotion}</p>
                    <p className="text-sm text-cyan-200">Confidence: {realtimeResult.confidencePercent.toFixed(1)}%</p>
                  </div>

                  <div className="space-y-2">
                    {realtimeProbabilityEntries.map(([emotion, score]) => (
                      <div key={`live-${emotion}`}>
                        <div className="flex items-center justify-between text-xs text-slate-300">
                          <span className="capitalize">{emotion}</span>
                          <span>{(score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="mt-1 h-2 rounded-full bg-white/10">
                          <div
                            className="h-2 rounded-full bg-gradient-to-r from-blue-500 via-violet-500 to-cyan-400"
                            style={{ width: `${Math.max(2, Math.min(100, score * 100))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-400">Start camera and real-time mode to stream predictions.</p>
              )}
            </div>
          </div>
        </Card>
      </div>
    </section>
  );
}
