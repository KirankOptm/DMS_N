#!/usr/bin/env python3
"""
Trigger MediaPipe to download face landmark model
Then find where it's cached
"""
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image

print("MediaPipe will download face landmark model on first use...")
print("This may take a moment...\n")

# Create a dummy image
dummy_image = np.zeros((192, 192, 3), dtype=np.uint8)
pil_image = Image.fromarray(dummy_image)

# Download model by creating FaceLandmarker
# This will cache the model to disk
try:
    # MediaPipe downloads models to cache directory
    print("Initializing FaceLandmarker (will download model)...")
    
    base_options = python.BaseOptions(model_asset_path='')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1)
    
    # This will fail but trigger model download
    detector = vision.FaceLandmarker.create_from_options(options)
    
except Exception as e:
    print(f"Expected error: {e}\n")

# Now search for cached models
print("Searching for cached models...")
print(f"\nChecking common cache locations:\n")

# Check MediaPipe cache directories
cache_dirs = [
    os.path.expanduser("~/.cache/mediapipe"),
    os.path.expanduser("~/AppData/Local/mediapipe"),
    os.path.expanduser("~/AppData/Roaming/mediapipe"),
    os.path.join(os.path.dirname(mp.__file__), "modules"),
    os.path.join(os.path.dirname(mp.__file__), "models"),
]

for cache_dir in cache_dirs:
    print(f"Checking: {cache_dir}")
    if os.path.exists(cache_dir):
        print(f"  ✓ Directory exists")
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith(('.tflite', '.task')):
                    full_path = os.path.join(root, file)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"    Found: {file} ({size_mb:.2f} MB)")
                    print(f"    Path: {full_path}")
    else:
        print(f"  ✗ Not found")
    print()

print("\n" + "="*60)
print("If no models found, MediaPipe uses bundled resources.")
print("Alternative: Use the .task file you already downloaded")
print("="*60)
