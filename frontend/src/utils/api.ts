export type EmotionProbabilities = Record<string, number>;

export interface FaceBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FacePrediction {
  id: number;
  bbox: FaceBox;
  emotion: string;
  confidence: number;
  probabilities: number[];
  probabilityMap: EmotionProbabilities;
}

export interface PredictionResponse {
  emotion: string;
  confidence: number;
  probabilities: EmotionProbabilities;
  boxes: FaceBox[];
  faces_predictions: FacePrediction[];
}

export interface PredictEmotionResponse {
  emotion: string;
  confidence: number;
  confidencePercent: number;
  allProbabilities: EmotionProbabilities;
  faces: FacePrediction[];
  boxes: FaceBox[];
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

function parsePredictPayload(payload: Record<string, unknown>): PredictEmotionResponse {
  const confidence = normalizeConfidence(payload.confidence ?? payload.score ?? payload.probability);
  const probabilities = normalizeProbabilityMap(
    payload.all_probabilities ?? payload.probabilities ?? payload.probabilityMap
  );

  const facesRaw = Array.isArray(payload.faces)
    ? payload.faces
    : Array.isArray(payload.faces_predictions)
      ? payload.faces_predictions
      : [];

  const faces: FacePrediction[] = facesRaw.map((item, index) => {
    const row = (item ?? {}) as Record<string, unknown>;
    const bboxRaw = (row.bbox ?? {}) as Record<string, unknown>;
    const faceProb = normalizeProbabilityMap(row.probabilityMap ?? row.all_probabilities ?? row.probabilities);

    return {
      id: Number(row.id ?? index + 1),
      bbox: {
        x: Number(bboxRaw.x ?? 0),
        y: Number(bboxRaw.y ?? 0),
        w: Number(bboxRaw.w ?? 0),
        h: Number(bboxRaw.h ?? 0),
      },
      emotion: String(row.emotion ?? "Unknown"),
      confidence: normalizeConfidence(row.confidence),
      probabilities: Object.values(faceProb),
      probabilityMap: faceProb,
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
    emotion: String(payload.emotion ?? payload.label ?? payload.predicted_emotion ?? "Unknown"),
    confidence,
    confidencePercent: confidence * 100,
    allProbabilities: probabilities,
    faces,
    boxes,
  };
}

export async function predictEmotion(file: File): Promise<PredictEmotionResponse> {
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

export async function predictRealtimeBase64(frameBase64: string): Promise<PredictEmotionResponse> {
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
  return backend;
}

export async function predictFrame(frameBlob: Blob): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("frame", frameBlob, "frame.jpg");

  const backend = await safeFetch<PredictionResponse>("/api/predict/realtime", {
    method: "POST",
    body: formData,
  });

  if (!backend) raiseBackendError("/api/predict/realtime");
  return backend;
}
