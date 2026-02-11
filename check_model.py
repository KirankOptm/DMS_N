import tensorflow as tf

model_path = r"D:\tflite_conversion\pfld_int8.tflite"
print("="*60)
print("Checking FaceLandmark INT8 model")
print("="*60)

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nINPUT:")
    inp = input_details[0]
    print(f"  Name: {inp.get('name', 'unnamed')}")
    print(f"  Shape: {inp['shape']}")
    print(f"  Dtype: {inp['dtype']}")
    print(f"  Quantization: scale={inp['quantization'][0]}, zero={inp['quantization'][1]}")
    
    print(f"\nOUTPUTS ({len(output_details)} total):")
    for i, out in enumerate(output_details):
        print(f"\n  [{i}] {out.get('name', 'unnamed')}")
        print(f"      Shape: {out['shape']}")
        print(f"      Dtype: {out['dtype']}")
        print(f"      Quantization: scale={out['quantization'][0]:.6f}, zero={out['quantization'][1]}")
    
    # Check if INT8
    is_int8_input = inp['dtype'] == tf.int8
    is_int8_outputs = all(out['dtype'] == tf.int8 for out in output_details)
    
    print("\n" + "="*60)
    if is_int8_input and is_int8_outputs:
        print("✓ FULLY INT8 - Can try Vela compilation directly!")
        print("\nNext step:")
        print('  python vela_compile.py "D:\\tflite_conversion\\410_FaceMeshV2\\face_landmarks_detector.tflite" "face_landmarks_detector_vela.tflite" ethos-u55-256')
    elif inp['dtype'] == tf.float32:
        print("✗ FLOAT32 model - Needs INT8 conversion")
        print("\nWe need to convert SavedModel to INT8")
    else:
        print(f"⚠ Mixed precision - Input: {inp['dtype']}, need to check compatibility")
        
except Exception as e:
    print(f"✗ Error loading model: {e}")