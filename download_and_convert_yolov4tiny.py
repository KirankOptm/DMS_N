#!/usr/bin/env python3
"""
Download YOLOv4-tiny Darknet weights and convert to TFLite INT8.
Then check Vela/NPU compatibility.

Steps:
  1. Download yolov4-tiny.weights + yolov4-tiny.cfg from Darknet
  2. Parse Darknet weights into a Keras model (fixed-size resize)
  3. Convert to TFLite INT8
  4. Compile with Vela
"""

import os
import sys
import struct
import urllib.request
import numpy as np

# ---- CONFIG ----
INPUT_SIZE = 416  # YOLOv4-tiny standard input
NUM_CLASSES = 80  # COCO classes
WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
WEIGHTS_FILE = "yolov4-tiny.weights"
CFG_FILE = "yolov4-tiny.cfg"
OUTPUT_TFLITE = "yolov4_tiny_coco_416_int8.tflite"


def download_file(url, filename):
    """Download file if not exists"""
    if os.path.isfile(filename):
        print(f"  [SKIP] {filename} already exists ({os.path.getsize(filename)/1024/1024:.1f} MB)")
        return
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"  [OK] {filename} ({os.path.getsize(filename)/1024/1024:.1f} MB)")


def parse_cfg(cfg_file):
    """Parse Darknet .cfg file into list of layer dicts"""
    with open(cfg_file, 'r') as f:
        lines = f.read().strip().split('\n')
    
    blocks = []
    block = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            if block:
                blocks.append(block)
            block = {'type': line[1:-1].strip()}
        else:
            key, val = line.split('=', 1)
            block[key.strip()] = val.strip()
    if block:
        blocks.append(block)
    return blocks


def build_yolov4_tiny_keras(input_size=416, num_classes=80):
    """
    Build YOLOv4-tiny in Keras with FIXED resize (no dynamic UpSampling2D).
    This ensures proper Vela compilation.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(input_size, input_size, 3), name='input')
    
    # ---- CSPDarknet53-tiny backbone ----
    # Conv0: 3x3, 32 filters, stride 2
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name='conv_0')(inputs)
    x = layers.BatchNormalization(name='bn_0')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_0')(x)
    
    # Conv1: 3x3, 64 filters, stride 2
    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False, name='conv_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_1')(x)
    
    # CSP Block 1
    # Route split - take second half of channels
    route_1 = layers.Lambda(lambda t: t[..., 32:], name='route_split_1')(x)
    
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv_2')(route_1)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_2')(x)
    
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_3')(x)
    
    x = layers.Concatenate(name='concat_0')([x, route_1])
    
    x = layers.Conv2D(64, 1, padding='same', use_bias=False, name='conv_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_4')(x)
    
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool_0')(x)
    
    # CSP Block 2
    route_2 = layers.Lambda(lambda t: t[..., 32:], name='route_split_2')(x)
    
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv_5')(route_2)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_5')(x)
    
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv_6')(x)
    x = layers.BatchNormalization(name='bn_6')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_6')(x)
    
    x = layers.Concatenate(name='concat_1')([x, route_2])
    
    x = layers.Conv2D(128, 1, padding='same', use_bias=False, name='conv_7')(x)
    x = layers.BatchNormalization(name='bn_7')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_7')(x)
    
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool_1')(x)
    
    # CSP Block 3
    route_3 = layers.Lambda(lambda t: t[..., 64:], name='route_split_3')(x)
    
    x = layers.Conv2D(64, 3, padding='same', use_bias=False, name='conv_8')(route_3)
    x = layers.BatchNormalization(name='bn_8')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_8')(x)
    
    x = layers.Conv2D(64, 3, padding='same', use_bias=False, name='conv_9')(x)
    x = layers.BatchNormalization(name='bn_9')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_9')(x)
    
    x = layers.Concatenate(name='concat_2')([x, route_3])
    
    x = layers.Conv2D(256, 1, padding='same', use_bias=False, name='conv_10')(x)
    x = layers.BatchNormalization(name='bn_10')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_10')(x)
    
    route_backbone = x  # Save for FPN (26x26x256)
    
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool_2')(x)
    
    # Final backbone conv
    x = layers.Conv2D(512, 3, padding='same', use_bias=False, name='conv_11')(x)
    x = layers.BatchNormalization(name='bn_11')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_11')(x)
    
    # ---- Neck / FPN ----
    # 1x1 conv to reduce channels
    x = layers.Conv2D(256, 1, padding='same', use_bias=False, name='conv_12')(x)
    x = layers.BatchNormalization(name='bn_12')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='leaky_12')(x)
    
    # Large object detection head (13x13)
    large_obj = layers.Conv2D(512, 3, padding='same', use_bias=False, name='conv_large')(x)
    large_obj = layers.BatchNormalization(name='bn_large')(large_obj)
    large_obj = layers.LeakyReLU(negative_slope=0.1, name='leaky_large')(large_obj)
    # YOLO output: 3 anchors * (5 + num_classes)
    large_output = layers.Conv2D(3 * (5 + num_classes), 1, padding='same', 
                                  use_bias=True, name='yolo_large')(large_obj)
    
    # Upsample for small object detection
    # *** FIXED RESIZE instead of dynamic UpSampling2D ***
    target_h = input_size // 16  # 26 for 416 input
    target_w = input_size // 16
    upsample = layers.Conv2D(128, 1, padding='same', use_bias=False, name='conv_upsample')(x)
    upsample = layers.BatchNormalization(name='bn_upsample')(upsample)
    upsample = layers.LeakyReLU(negative_slope=0.1, name='leaky_upsample')(upsample)
    # Use tf.image.resize with FIXED size — this converts to RESIZE_BILINEAR with known dims
    upsample = layers.Resizing(target_h, target_w, interpolation='nearest', name='resize_fixed')(upsample)
    
    # Concatenate with backbone feature map
    x = layers.Concatenate(name='concat_fpn')([upsample, route_backbone])
    
    # Small object detection head (26x26)
    small_obj = layers.Conv2D(256, 3, padding='same', use_bias=False, name='conv_small')(x)
    small_obj = layers.BatchNormalization(name='bn_small')(small_obj)
    small_obj = layers.LeakyReLU(negative_slope=0.1, name='leaky_small')(small_obj)
    small_output = layers.Conv2D(3 * (5 + num_classes), 1, padding='same',
                                  use_bias=True, name='yolo_small')(small_obj)
    
    model = keras.Model(inputs=inputs, outputs=[large_output, small_output], name='yolov4_tiny')
    return model


def load_darknet_weights(model, weights_file, cfg_blocks):
    """Load Darknet .weights into Keras model"""
    with open(weights_file, 'rb') as f:
        # Header: 5 int32 values (major, minor, revision, seen_images[2])
        major, minor, revision = struct.unpack('3i', f.read(12))
        if (major * 10 + minor) >= 2:
            seen = struct.unpack('Q', f.read(8))[0]
        else:
            seen = struct.unpack('I', f.read(4))[0]
        print(f"  Darknet version: {major}.{minor}.{revision}, seen: {seen}")
        
        weights_data = np.fromfile(f, dtype=np.float32)
    
    print(f"  Total weight values: {len(weights_data)}")
    
    # Map weights to conv layers in order
    ptr = 0
    conv_layers = [l for l in model.layers if 'conv' in l.name.lower() and hasattr(l, 'kernel')]
    bn_layers = [l for l in model.layers if 'bn' in l.name.lower()]
    
    conv_idx = 0
    bn_idx = 0
    
    for block in cfg_blocks:
        if block['type'] != 'convolutional':
            continue
        
        if conv_idx >= len(conv_layers):
            break
            
        conv = conv_layers[conv_idx]
        use_bn = int(block.get('batch_normalize', 0))
        
        filters = conv.kernel.shape[-1]
        kernel_size = conv.kernel.shape[0]
        in_channels = conv.kernel.shape[2]
        
        if use_bn and bn_idx < len(bn_layers):
            bn = bn_layers[bn_idx]
            bn_size = filters
            
            # BN: beta, gamma, mean, var
            beta = weights_data[ptr:ptr + bn_size]; ptr += bn_size
            gamma = weights_data[ptr:ptr + bn_size]; ptr += bn_size
            mean = weights_data[ptr:ptr + bn_size]; ptr += bn_size
            var = weights_data[ptr:ptr + bn_size]; ptr += bn_size
            
            bn.set_weights([gamma, beta, mean, var])
            bn_idx += 1
        else:
            # No BN, load bias
            bias = weights_data[ptr:ptr + filters]; ptr += filters
            
        # Conv weights: Darknet = [out, in, h, w], Keras = [h, w, in, out]
        num_weights = filters * in_channels * kernel_size * kernel_size
        conv_weights = weights_data[ptr:ptr + num_weights]; ptr += num_weights
        conv_weights = conv_weights.reshape(filters, in_channels, kernel_size, kernel_size)
        conv_weights = conv_weights.transpose(2, 3, 1, 0)  # [h, w, in, out]
        
        if use_bn:
            conv.set_weights([conv_weights])
        else:
            conv.set_weights([conv_weights, bias])
        
        conv_idx += 1
    
    print(f"  Loaded {conv_idx} conv layers, {bn_idx} BN layers")
    print(f"  Weight pointer: {ptr}/{len(weights_data)} ({ptr/len(weights_data)*100:.1f}%)")
    return model


def convert_to_tflite_int8(model, output_path, input_size):
    """Convert Keras model to INT8 TFLite"""
    import tensorflow as tf
    
    # Representative dataset for quantization calibration
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, input_size, input_size, 3).astype(np.float32)
            yield [data]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("  Converting to INT8 TFLite (this may take a minute)...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"  [OK] Saved: {output_path} ({len(tflite_model)/1024/1024:.1f} MB)")
    return output_path


def check_vela_compatibility(tflite_path):
    """Quick check: compile with Vela and see results"""
    import subprocess
    
    print(f"\n  Running Vela on {tflite_path}...")
    result = subprocess.run(
        ['vela', '--accelerator-config', 'ethos-u65-256', tflite_path,
         '--output-dir', '.'],
        capture_output=True, text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0:
        print("  [OK] Vela compilation PASSED!")
    else:
        print("  [FAIL] Vela compilation failed")
    
    return result.returncode == 0


if __name__ == '__main__':
    print("=" * 60)
    print("  YOLOv4-tiny: Download → Convert → NPU Check")
    print("=" * 60)
    
    # Step 1: Download
    print("\n[1/5] Downloading Darknet files...")
    download_file(WEIGHTS_URL, WEIGHTS_FILE)
    download_file(CFG_URL, CFG_FILE)
    
    # Step 2: Parse config
    print("\n[2/5] Parsing config...")
    cfg_blocks = parse_cfg(CFG_FILE)
    conv_count = sum(1 for b in cfg_blocks if b['type'] == 'convolutional')
    print(f"  Config has {len(cfg_blocks)} blocks, {conv_count} conv layers")
    
    # Step 3: Build Keras model with fixed resize
    print("\n[3/5] Building Keras model (fixed resize for NPU)...")
    model = build_yolov4_tiny_keras(INPUT_SIZE, NUM_CLASSES)
    model.summary(line_length=100, print_fn=lambda x: print(f"  {x}"))
    
    # Step 4: Load Darknet weights
    print("\n[4/5] Loading Darknet weights...")
    model = load_darknet_weights(model, WEIGHTS_FILE, cfg_blocks)
    
    # Step 5: Convert to INT8 TFLite
    print("\n[5/5] Converting to INT8 TFLite...")
    convert_to_tflite_int8(model, OUTPUT_TFLITE, INPUT_SIZE)
    
    # Step 6: Check Vela
    print("\n[BONUS] Checking Vela NPU compatibility...")
    check_vela_compatibility(OUTPUT_TFLITE)
    
    print("\n[DONE] Complete!")
