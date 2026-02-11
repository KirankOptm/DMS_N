#!/usr/bin/env python3
"""
Vela compiler helper for NXP i.MX / Ethos-U NPU optimization.
Converts a TFLite int8 model to a Vela-optimized .tflite for NPU acceleration.

Requirements:
  pip install ethos-u-vela

Usage:
  python vela_compile.py models/detect.tflite models/detect_vela.tflite
"""
import sys
import os

def compile_with_vela(input_tflite: str, output_tflite: str, accelerator: str = "ethos-u65-256"):
    """
    Run Vela compiler on a TFLite model.
    
    Args:
        input_tflite: Path to input int8 quantized .tflite
        output_tflite: Path to output Vela-optimized .tflite
        accelerator: Target NPU config (ethos-u55-128, ethos-u55-256, ethos-u65-256, ethos-u65-512)
    """
    if not os.path.isfile(input_tflite):
        print(f"Error: Input file not found: {input_tflite}")
        sys.exit(1)
    
    try:
        # Try newer import path first (vela 3.x)
        try:
            from ethosu.vela import vela as vela_main
        except ImportError:
            # Fallback to older import (vela 2.x)
            import vela.vela as vela_main
    except ImportError:
        print("Error: ethos-u-vela not installed.")
        print("Install with: pip install ethos-u-vela")
        sys.exit(1)
    
    # Vela CLI args for i.MX93 NPU
    # NPU optimization for Conv2D, DepthwiseConv2D, ReLU, etc.
    vela_args = [
        input_tflite,
        "--output-dir", os.path.dirname(output_tflite) or ".",
        "--accelerator-config", accelerator,
        "--config", "imx93_vela_config.ini",
        "--system-config", "Ethos_U65_High_End_Embedded",
        "--memory-mode", "Shared_Sram",
        "--optimise", "Performance",  # Optimize for speed (alternative: "Size")
    ]
    
    print(f"Running Vela compiler:")
    print(f"  Input:  {input_tflite}")
    print(f"  Output: {output_tflite}")
    print(f"  Accelerator: {accelerator}")
    print(f"  Args: {' '.join(vela_args)}")
    
    # Vela writes output with a suffix; rename to desired path
    try:
        sys.argv = ["vela"] + vela_args  # Vela reads sys.argv
        vela_main.main()
        
        # Vela typically appends _vela.tflite; find and rename
        out_dir = os.path.dirname(output_tflite) or "."
        base = os.path.basename(input_tflite).replace(".tflite", "")
        vela_output = os.path.join(out_dir, f"{base}_vela.tflite")
        
        if os.path.isfile(vela_output) and vela_output != output_tflite:
            os.rename(vela_output, output_tflite)
            print(f"✓ Vela-optimized model saved to: {output_tflite}")
        elif os.path.isfile(output_tflite):
            print(f"✓ Vela-optimized model: {output_tflite}")
        else:
            print(f"Warning: Expected output not found. Check {out_dir} for *_vela.tflite")
    except Exception as e:
        print(f"Vela compilation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vela_compile.py <input.tflite> <output.tflite> [accelerator]")
        print("  accelerator options: ethos-u55-128, ethos-u55-256, ethos-u65-256 (default), ethos-u65-512")
        sys.exit(1)
    
    inp = sys.argv[1]
    out = sys.argv[2]
    accel = sys.argv[3] if len(sys.argv) > 3 else "ethos-u65-256"
    
    compile_with_vela(inp, out, accel)
