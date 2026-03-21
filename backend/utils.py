from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import HTTPException


def decode_upload_bytes(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return frame


def decode_base64_image(payload: str) -> np.ndarray:
    data = payload.strip()
    if "," in data and "base64" in data:
        data = data.split(",", 1)[1]

    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 frame payload") from exc

    return decode_upload_bytes(raw)
