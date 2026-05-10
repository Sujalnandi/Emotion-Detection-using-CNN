#!/usr/bin/env python3
"""Verify that the Python environment can import the project dependencies."""

from __future__ import annotations

import os
import sys

print("=" * 72)
print("FACIAL EMOTION RECOGNITION ENVIRONMENT CHECK")
print("=" * 72)
print()

print(f"Python: {sys.version.split()[0]}")
print(f"Executable: {sys.executable}")
print()

checks = []

try:
    import cv2
    checks.append(("cv2", cv2.__version__))
except Exception as exc:
    print(f"FAILED: cv2 import -> {exc}")
    raise SystemExit(1)

try:
    import mediapipe as mp
    checks.append(("mediapipe", getattr(mp, "__version__", "unknown")))
except Exception as exc:
    print(f"FAILED: mediapipe import -> {exc}")
    raise SystemExit(1)

try:
    import mtcnn
    checks.append(("mtcnn", getattr(mtcnn, "__version__", "unknown")))
except Exception as exc:
    print(f"FAILED: mtcnn import -> {exc}")
    raise SystemExit(1)

try:
    import tensorflow as tf
    checks.append(("tensorflow", tf.__version__))
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import load_model
    _ = Dense
    _ = load_model
except Exception as exc:
    print(f"FAILED: tensorflow / keras import -> {exc}")
    raise SystemExit(1)

try:
    from backend.config import HAAR_CASCADE_PATH
    checks.append(("haar_cascade_path", HAAR_CASCADE_PATH))
except Exception as exc:
    print(f"FAILED: backend.config import -> {exc}")
    raise SystemExit(1)

print("Imports OK:")
for name, value in checks:
    print(f"- {name}: {value}")

print()
print("Haar cascade path:")
print(HAAR_CASCADE_PATH)
print()
print("Environment check passed.")
