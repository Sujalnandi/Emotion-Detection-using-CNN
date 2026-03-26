#!/usr/bin/env python3
"""Test script to verify all TensorFlow imports work correctly"""

import sys
import numpy as np

print("=" * 60)
print("TENSORFLOW IMPORT VERIFICATION TEST")
print("=" * 60)
print()

# Test 1: TensorFlow Basic
print("✓ Test 1: Basic TensorFlow Import")
try:
    import tensorflow as tf
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  Keras version: {tf.keras.__version__}")
except ImportError as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Keras Preprocessing
print("\n✓ Test 2: Keras Preprocessing")
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print(f"  ImageDataGenerator imported successfully")
except ImportError as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Keras Applications
print("\n✓ Test 3: Keras Applications (EfficientNet)")
try:
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
    print(f"  EfficientNetB0 imported successfully")
except ImportError as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 4: Keras Callbacks
print("\n✓ Test 4: Keras Callbacks")
try:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
    print(f"  Callbacks imported successfully")
except ImportError as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 5: Additional Required Imports
print("\n✓ Test 5: Additional Dependencies")
try:
    import cv2
    import numpy
    import sklearn
    from matplotlib import pyplot
    print(f"  cv2, numpy, sklearn, matplotlib all working")
except ImportError as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 6: Device Configuration
print("\n✓ Test 6: TensorFlow Device Configuration")
try:
    devices = tf.config.list_physical_devices()
    print(f"  Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i+1}: {device}")
except Exception as e:
    print(f"  ⚠ Warning: {e}")

# Test 7: Simple Model Creation
print("\n✓ Test 7: Model Creation Test")
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(28,)),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    print(f"  Model created successfully")
    print(f"  Model parameters: {model.count_params():,}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("✅ ALL TESTS PASSED - YOUR SETUP IS WORKING!")
print("=" * 60)
print()
print("Next steps:")
print("1. Reload VS Code (Ctrl+Shift+P → Developer: Reload Window)")
print("2. Errors should disappear from Pylance")
print("3. You're ready to run your emotion detection project!")
