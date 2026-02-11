#!/usr/bin/env python3
"""
Check TFLite model quantization status and Vela compatibility
"""
import sys
import os

def check_model(model_path):
    print(f"Checking model: {model_path}\n")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    # Get file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    
    try:
        import tensorflow as tf
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n" + "="*60)
        print("INPUT DETAILS")
        print("="*60)
        for i, inp in enumerate(input_details):
            print(f"\nInput #{i}:")
            print(f"  Name: {inp['name']}")
            print(f"  Shape: {inp['shape']}")
            print(f"  Dtype: {inp['dtype']}")
            
            # Check quantization
            quant_params = inp.get('quantization_parameters', {})
            if quant_params:
                scales = quant_params.get('scales', [])
                zero_points = quant_params.get('zero_points', [])
                if len(scales) > 0 and scales[0] != 0:
                    print(f"  Quantized: YES (INT8)")
                    print(f"    Scale: {scales[0]}")
                    print(f"    Zero point: {zero_points[0] if zero_points else 'N/A'}")
                else:
                    print(f"  Quantized: NO (FLOAT32)")
            else:
                print(f"  Quantized: NO (FLOAT32)")
        
        print("\n" + "="*60)
        print("OUTPUT DETAILS")
        print("="*60)
        for i, out in enumerate(output_details):
            print(f"\nOutput #{i}:")
            print(f"  Name: {out['name']}")
            print(f"  Shape: {out['shape']}")
            print(f"  Dtype: {out['dtype']}")
            
            # Check quantization
            quant_params = out.get('quantization_parameters', {})
            if quant_params:
                scales = quant_params.get('scales', [])
                zero_points = quant_params.get('zero_points', [])
                if len(scales) > 0 and scales[0] != 0:
                    print(f"  Quantized: YES (INT8)")
                    print(f"    Scale: {scales[0]}")
                    print(f"    Zero point: {zero_points[0] if zero_points else 'N/A'}")
                else:
                    print(f"  Quantized: NO (FLOAT32)")
            else:
                print(f"  Quantized: NO (FLOAT32)")
        
        # Check ALL internal tensors
        print("\n" + "="*60)
        print("INTERNAL TENSOR ANALYSIS")
        print("="*60)
        all_tensors = interpreter.get_tensor_details()
        dtype_counts = {}
        int8_count = 0
        float32_count = 0
        other_count = 0
        for t in all_tensors:
            dtype_str = str(t['dtype'])
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            if t['dtype'] in [tf.int8, tf.uint8]:
                int8_count += 1
            elif t['dtype'] == tf.float32:
                float32_count += 1
            else:
                other_count += 1
        
        print(f"\nTotal tensors: {len(all_tensors)}")
        for dtype_str, count in sorted(dtype_counts.items()):
            print(f"  {dtype_str}: {count} tensors")
        
        # Show first few float32 tensors if they exist
        float32_tensors = [t for t in all_tensors if t['dtype'] == tf.float32]
        if float32_tensors:
            print(f"\nFloat32 tensors ({len(float32_tensors)} total):")
            for t in float32_tensors[:10]:
                is_io = "(INPUT/OUTPUT)" if any(t['index'] == inp['index'] for inp in input_details + output_details) else "(INTERNAL)"
                print(f"  [{t['index']:3d}] {t['name'][:60]:60s} shape={t['shape']} {is_io}")
            if len(float32_tensors) > 10:
                print(f"  ... and {len(float32_tensors) - 10} more")
        
        # Determine model type
        print("\n" + "="*60)
        print("MODEL ANALYSIS")
        print("="*60)
        
        input_dtype = input_details[0]['dtype']
        io_is_float = input_dtype == tf.float32
        internals_mostly_int8 = int8_count > float32_count
        
        # Count float32 that are NOT I/O
        io_indices = set(d['index'] for d in input_details + output_details)
        internal_float32 = [t for t in all_tensors if t['dtype'] == tf.float32 and t['index'] not in io_indices]
        internal_int8 = [t for t in all_tensors if t['dtype'] in [tf.int8, tf.uint8] and t['index'] not in io_indices]
        
        if io_is_float and len(internal_int8) > len(internal_float32):
            pct = len(internal_int8) / (len(all_tensors) - len(io_indices)) * 100
            print(f"\n✓ Model is INT8 QUANTIZED internally ({pct:.1f}% int8 tensors)")
            print(f"  BUT input/output are FLOAT32 (quantize/dequantize wrappers)")
            print(f"  Internal INT8 tensors: {len(internal_int8)}")
            print(f"  Internal FLOAT32 tensors: {len(internal_float32)}")
            print(f"\n  → This means the model has QUANTIZE + DEQUANTIZE nodes")
            print(f"    at the edges. Vela will delegate INT8 ops to NPU,")
            print(f"    quant/dequant run on CPU.")
            print(f"\n  → For Ethos-U65 Vela compilation: SHOULD WORK")
            print(f"    Vela handles float I/O with internal int8 correctly.")
        elif not io_is_float:
            print(f"\n✓ Model is FULLY INT8 quantized (I/O + internal)")
            print("✓ Ready for Vela compilation")
        else:
            print(f"\n✗ Model is FLOAT32 (not quantized)")
            print(f"  Internal INT8: {len(internal_int8)}, Internal FLOAT32: {len(internal_float32)}")
            print("✗ Needs INT8 quantization BEFORE Vela compilation")
        
    except ImportError:
        print("\nERROR: TensorFlow not installed")
        print("Install: py -m pip install tensorflow")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "D:\\old_laptop_data\\DMS_with_yolo(npu)\\hand_landmark_full_quant.tflite"
    
    check_model(model_path)
