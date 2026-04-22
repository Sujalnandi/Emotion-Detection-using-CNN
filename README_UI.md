# Facial Emotion Detection System - AI Dashboard UI

This frontend is a futuristic React + TypeScript dashboard for the Facial Emotion Detection System backend.

## Tech Stack

- React
- TypeScript
- Tailwind CSS
- Framer Motion
- Lucide Icons
- React Router
- Vite

## Backend API Integration

Set backend URL using environment variable:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Used endpoints:

- `POST /api/predict/image` (multipart form field: `image`)
- `POST /api/predict/realtime` (multipart form field: `frame`)

Expected response shape:

```json
{
  "emotion": "Happy",
  "confidence": 0.92,
  "probabilities": {
    "Angry": 0.01,
    "Disgust": 0.01,
    "Fear": 0.01,
    "Happy": 0.92,
    "Neutral": 0.03,
    "Sad": 0.01,
    "Surprise": 0.01
  },
  "boxes": [
    { "x": 0.3, "y": 0.2, "w": 0.35, "h": 0.5 }
  ]
}
```

`boxes` supports either normalized coordinates (`0-1`) or percentage-like values.

## Install and Run

```bash
npm install
npm run dev
```

## Start Shared Inference Backend

Run this in a second terminal from `facial_emotion_detection` directory:

```bash
cd facial_emotion_detection
python -m uvicorn emotion_detection.ui_app:app --reload --port 8000
```

This backend uses the same shared pipeline as webcam realtime detection via:

- `emotion_detection/inference_engine.py`
- `predict_frame()`
- `EmotionPredictor`

This guarantees UI and realtime script produce identical predictions for the same frame/image.

## Tailwind Notes

Tailwind is configured via:

- `tailwind.config.ts`
- `postcss.config.js`
- `src/index.css`
