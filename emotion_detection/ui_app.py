"""Compatibility wrapper for legacy import path.

Allows running:
    python -m uvicorn emotion_detection.ui_app:app --reload --port 8000
"""

from backend.main import app

__all__ = ["app"]
