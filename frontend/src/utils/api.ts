export type EmotionProbabilities = Record<string, number>;

export interface RankedPrediction {
  emotion: string;
  confidence: number;
}

export interface FaceBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FacePrediction {
  id: number;
  box?: [number, number, number, number];
  bbox: FaceBox;
  emotion: string;
  displayLabel?: string;
  confidence: number;
  confidenceLevel?: "high" | "medium" | "low";
  lowConfidence?: boolean;
  probabilities: EmotionProbabilities;
  top3?: RankedPrediction[];
}

export interface PredictionResponse {
  emotion: string;
  displayLabel?: string;
  confidence: number;
  confidenceLevel?: "high" | "medium" | "low";
  lowConfidence?: boolean;
  probabilities: EmotionProbabilities;
  top3?: RankedPrediction[];
  boxes: FaceBox[];
  faces: FacePrediction[];
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function safeFetch<T>(endpoint: string, options: RequestInit): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, options);
    if (!res.ok) {
      return null;
    }
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

function raiseBackendError(endpoint: string): never {
  throw new Error(
    `Prediction failed at ${endpoint}. Start backend API: python -m uvicorn emotion_detection.ui_app:app --reload --port 8000`
  );
}

function normalizeConfidence(value: unknown): number {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  if (value > 1) return Math.min(100, value) / 100;
  return Math.max(0, value);
}

function normalizeProbabilityMap(raw: unknown): EmotionProbabilities {
  if (!raw || typeof raw !== "object") return {};

  const entries = Object.entries(raw as Record<string, unknown>).map(([k, v]) => {
    const num = typeof v === "number" && !Number.isNaN(v) ? v : 0;
    const normalized = num > 1 ? Math.min(100, num) / 100 : Math.max(0, num);
    return [k, normalized] as const;
  });

  return Object.fromEntries(entries);
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function confidenceLevel(confidence: number): "high" | "medium" | "low" {
  if (confidence < 0.4) return "low";
  if (confidence < 0.7) return "medium";
  return "high";
}

function formatDisplayLabel(emotion: string, confidence: number): string {
  const readable = emotion.charAt(0).toUpperCase() + emotion.slice(1).toLowerCase();
  if (confidence < 0.4) {
    return `${readable} (low confidence ${Math.round(confidence * 100)}%)`;
  }
  return readable;
}

function deriveTop3(probabilities: EmotionProbabilities): RankedPrediction[] {
  return Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([emotion, confidence]) => ({ emotion, confidence }));
}

function parseTop3(raw: unknown, fallback: EmotionProbabilities): RankedPrediction[] {
  if (Array.isArray(raw)) {
    const parsed = raw
      .map((entry) => {
        const item = (entry ?? {}) as Record<string, unknown>;
        return {
          emotion: String(item.emotion ?? "unknown"),
          confidence: normalizeConfidence(item.confidence),
        };
      })
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    if (parsed.length > 0) return parsed;
  }

  return deriveTop3(fallback);
}

function parsePredictPayload(payload: Record<string, unknown>): PredictionResponse {
  const confidence = normalizeConfidence(payload.confidence ?? payload.score ?? payload.probability);
  const probabilities = normalizeProbabilityMap(payload.probabilities ?? payload.all_probabilities);
  const top3 = parseTop3(payload.top3, probabilities);
  const emotion = String(payload.emotion ?? payload.label ?? payload.predicted_emotion ?? "unknown").toLowerCase();
  const parsedConfidenceLevel = String(payload.confidence_level ?? "").toLowerCase();
  const resolvedConfidenceLevel =
    parsedConfidenceLevel === "high" || parsedConfidenceLevel === "medium" || parsedConfidenceLevel === "low"
      ? parsedConfidenceLevel
      : confidenceLevel(confidence);

  const facesRaw = Array.isArray(payload.faces)
    ? payload.faces
    : Array.isArray(payload.faces_predictions)
      ? payload.faces_predictions
      : [];

  const faces: FacePrediction[] = facesRaw.map((item, index) => {
    const row = (item ?? {}) as Record<string, unknown>;
    const bboxRaw = (row.bbox ?? {}) as Record<string, unknown>;
    const boxRaw = Array.isArray(row.box) ? row.box : null;
    const faceProb = normalizeProbabilityMap(row.probabilityMap ?? row.all_probabilities ?? row.probabilities);
    const faceEmotion = String(row.emotion ?? "unknown").toLowerCase();
    const faceConfidence = normalizeConfidence(row.confidence);
    const faceTop3 = parseTop3(row.top3, faceProb);
    const faceConfidenceLevelRaw = String(row.confidence_level ?? "").toLowerCase();
    const faceConfidenceLevel =
      faceConfidenceLevelRaw === "high" || faceConfidenceLevelRaw === "medium" || faceConfidenceLevelRaw === "low"
        ? faceConfidenceLevelRaw
        : confidenceLevel(faceConfidence);

    const bbox = {
      x: clamp01(Number(bboxRaw.x ?? 0)),
      y: clamp01(Number(bboxRaw.y ?? 0)),
      w: clamp01(Number(bboxRaw.w ?? 0)),
      h: clamp01(Number(bboxRaw.h ?? 0)),
    };

    const parsedBox: [number, number, number, number] | undefined =
      boxRaw && boxRaw.length >= 4
        ? [Number(boxRaw[0] ?? 0), Number(boxRaw[1] ?? 0), Number(boxRaw[2] ?? 0), Number(boxRaw[3] ?? 0)]
        : undefined;

    return {
      id: Number(row.id ?? index + 1),
      box: parsedBox,
      bbox,
      emotion: faceEmotion,
      displayLabel: String(row.display_label ?? formatDisplayLabel(faceEmotion, faceConfidence)),
      confidence: faceConfidence,
      confidenceLevel: faceConfidenceLevel,
      lowConfidence: Boolean(row.low_confidence ?? faceConfidence < 0.4),
      probabilities: faceProb,
      top3: faceTop3,
    };
  });

  const boxesRaw = Array.isArray(payload.boxes) ? payload.boxes : [];
  const boxes: FaceBox[] = boxesRaw.map((b) => {
    const box = (b ?? {}) as Record<string, unknown>;
    return {
      x: Number(box.x ?? 0),
      y: Number(box.y ?? 0),
      w: Number(box.w ?? 0),
      h: Number(box.h ?? 0),
    };
  });

  return {
    emotion,
    displayLabel: String(payload.display_label ?? formatDisplayLabel(emotion, confidence)),
    confidence,
    confidenceLevel: resolvedConfidenceLevel,
    lowConfidence: Boolean(payload.low_confidence ?? confidence < 0.4),
    probabilities,
    top3,
    faces,
    boxes,
  };
}

export async function predictEmotion(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });
  } catch {
    raiseBackendError("/predict");
  }

  if (!res.ok) {
    const details = await res.text().catch(() => "");
    throw new Error(details || `API request failed with status ${res.status}.`);
  }

  const payload = (await res.json()) as Record<string, unknown>;
  return parsePredictPayload(payload);
}

export async function predictRealtimeBase64(frameBase64: string): Promise<PredictionResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/realtime`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: frameBase64 }),
    });
  } catch {
    raiseBackendError("/realtime");
  }

  if (!res.ok) {
    const details = await res.text().catch(() => "");
    throw new Error(details || `API request failed with status ${res.status}.`);
  }

  const payload = (await res.json()) as Record<string, unknown>;
  return parsePredictPayload(payload);
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const backend = await safeFetch<PredictionResponse>("/api/predict/image", {
    method: "POST",
    body: formData,
  });

  if (!backend) raiseBackendError("/api/predict/image");
  return parsePredictPayload(backend as unknown as Record<string, unknown>);
}

export async function predictFrame(frameBlob: Blob): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("frame", frameBlob, "frame.jpg");

  const backend = await safeFetch<PredictionResponse>("/api/predict/realtime", {
    method: "POST",
    body: formData,
  });

  if (!backend) raiseBackendError("/api/predict/realtime");
  return parsePredictPayload(backend as unknown as Record<string, unknown>);
}
