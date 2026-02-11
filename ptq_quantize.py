"""
PTQ Quantization script for hand_landmark and palm_detection models.
Converts SavedModel -> INT8 quantized TFLite (with float32 I/O).
Same method used for the working face detection / face landmark models.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import glob
import cv2
import sys

CALIB_DIR = "calib_hand"

def get_calib_images(target_size, max_images=200):
    """Load and resize calibration images."""
    paths = glob.glob(os.path.join(CALIB_DIR, "*.jpg")) + \
            glob.glob(os.path.join(CALIB_DIR, "*.png")) + \
            glob.glob(os.path.join(CALIB_DIR, "*.jpeg"))
    paths = paths[:max_images]
    print(f"  Using {len(paths)} calibration images, resizing to {target_size}x{target_size}")
    return paths, target_size

def quantize_model(saved_model_dir, output_name, input_size):
    """PTQ quantize a SavedModel to INT8 TFLite with float32 I/O."""
    print(f"\n{'='*60}")
    print(f"Quantizing: {saved_model_dir} -> {output_name}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"{'='*60}")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    paths, size = get_calib_images(input_size)

    def representative_dataset():
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            img = img.astype(np.float32) / 255.0
            yield [np.expand_dims(img, axis=0)]

    converter.representative_dataset = representative_dataset

    # INT8 internal ops, but keep float32 I/O (same as working face models)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    print("  Running quantization (this may take a few minutes)...")
    tflite_model = converter.convert()

    with open(output_name, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_name) / (1024 * 1024)
    print(f"  Saved: {output_name} ({size_mb:.2f} MB)")

    # Verify the model
    print("  Verifying...")
    interp = tf.lite.Interpreter(model_path=output_name)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    print(f"  Input: {inp['shape']} dtype={inp['dtype']}")
    for i, o in enumerate(outs):
        print(f"  Output {i}: {o['name']} {o['shape']} dtype={o['dtype']}")
    print("  OK!\n")


if __name__ == "__main__":
    # 1. Hand Landmark: input 224x224
    quantize_model(
        saved_model_dir="hand_landmark_lite_saved_model",
        output_name="hand_landmark_ptq.tflite",
        input_size=224
    )

    # 2. Palm Detection: input 192x192
    quantize_model(
        saved_model_dir="palm_detection_lite_saved_model",
        output_name="palm_detection_ptq.tflite",
        input_size=192
    )

    print("=" * 60)
    print("DONE! Both PTQ models created:")
    print("  - hand_landmark_ptq.tflite")
    print("  - palm_detection_ptq.tflite")
    print("\nNext step: Vela compile on the board:")
    print("  vela --accelerator-config ethos-u65-256 hand_landmark_ptq.tflite")
    print("  vela --accelerator-config ethos-u65-256 palm_detection_ptq.tflite")
    print("=" * 60)
