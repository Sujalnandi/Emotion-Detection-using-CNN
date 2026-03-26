from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import HTTPException


def decode_upload_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes from file upload.
    
    Args:
        file_bytes: Raw bytes from uploaded file
        
    Returns:
        BGR numpy array (OpenCV format)
        
    Raises:
        HTTPException: If image data is invalid or corrupted
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=400, 
            detail="Invalid image data: Could not decode image file. Ensure it's a valid image format (JPEG, PNG, etc.)"
        )
    return frame


def decode_base64_image(payload: str) -> np.ndarray:
    """Decode base64-encoded image string.
    
    Args:
        payload: Base64-encoded image string (with optional 'data:image/...;base64,' prefix)
        
    Returns:
        BGR numpy array (OpenCV format)
        
    Raises:
        HTTPException: If base64 is invalid or image cannot be decoded
    """
    data = payload.strip()
    if "," in data and "base64" in data:
        data = data.split(",", 1)[1]

    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400, 
            detail="Invalid base64 payload: Ensure the string is properly base64-encoded"
        ) from exc

    return decode_upload_bytes(raw)
