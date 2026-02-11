#!/usr/bin/env python3
"""
Convert FLOAT32 TFLite to INT8 using Post-Training Quantization (PTQ)
This is what NXP likely used for MediaPipe models

Unlike direct TFLite conversion, we need to:
1. Load the float model into TF Lite Interpreter
2. Run inference on representative data to collect statistics
3. Use TF Lite Converter with quantization
"""
import tensorflow as tf
import numpy as np
import os
import cv2

def load_calibration_images(calib_dir, target_size=(256, 256)):
    """Load and preprocess calibration images"""
    images = []
    
    print(f"Loading calibration images from: {calib_dir}")
    
    for img_file in os.listdir(calib_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(calib_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize to model input size
                img = cv2.resize(img, target_size)
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1] float32
                img = img.astype(np.float32) / 255.0
                images.append(img)
                
                if len(images) >= 100:  # Limit to 100 images
                    break
    
    print(f"Loaded {len(images)} calibration images")
    return images

def representative_dataset_gen(images):
    """Generator for representative dataset (required for PTQ)"""
    for img in images:
        # Add batch dimension
        yield [np.expand_dims(img, axis=0)]

def convert_float_to_int8_ptq(float_model_path, output_path, calib_images):
    """
    Post-Training Quantization (PTQ) - NXP Method
    Converts FLOAT32 TFLite to INT8 TFLite
    """
    print(f"\n{'='*60}")
    print("POST-TRAINING QUANTIZATION (PTQ)")
    print(f"{'='*60}\n")
    
    print(f"Input: {float_model_path}")
    print(f"Output: {output_path}\n")
    
    # Create converter from existing TFLite model
    # We need to use the experimental converter API
    try:
        # Load the float model
        interpreter = tf.lite.Interpreter(model_path=float_model_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Model Information:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Outputs: {len(output_details)}")
        
        # PTQ using TFLite Converter
        # NOTE: We cannot directly convert TFLite -> INT8 TFLite
        # This requires the original SavedModel or Keras model
        
        print("\n" + "="*60)
        print("CRITICAL LIMITATION")
        print("="*60)
        print("❌ Cannot convert TFLite FLOAT32 → TFLite INT8")
        print("✓ NEED original model format:")
        print("  - SavedModel (.pb)")
        print("  - Keras model (.h5)")
        print("  - TensorFlow frozen graph")
        print("\nNXP Solution:")
        print("  1. They have access to original MediaPipe source models")
        print("  2. Convert from source using tf.lite.TFLiteConverter")
        print("  3. Apply PTQ during initial conversion")
        print("\nYour Options:")
        print("  A. Find MediaPipe source models (.pb or .h5)")
        print("  B. Use existing INT8 models (pfld_int8.tflite)")
        print("  C. Train your own face landmark model")
        print("  D. Ask NXP for their INT8 MediaPipe models")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if face landmark model exists
    float_model = "extracted_task/face_landmarks_detector.tflite"
    calib_dir = "calib_fr"
    
    if not os.path.exists(float_model):
        print(f"ERROR: Model not found: {float_model}")
        print("Extract it first with: python extract_task_zip.py face_landmark_mediapipe.task")
        exit(1)
    
    if not os.path.exists(calib_dir):
        print(f"ERROR: Calibration directory not found: {calib_dir}")
        exit(1)
    
    # Load calibration images
    calib_images = load_calibration_images(calib_dir, target_size=(256, 256))
    
    # Attempt conversion
    convert_float_to_int8_ptq(float_model, "face_landmarks_int8.tflite", calib_images)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Search NXP repo for INT8 models or conversion scripts")
    print("2. Contact NXP for MediaPipe INT8 models")
    print("3. Or use alternative INT8 face landmark models")
