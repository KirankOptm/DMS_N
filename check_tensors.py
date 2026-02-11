
#!/usr/bin/env python3

from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL = "scrfd_500m_full_int8_vela.tflite"

print("\n===== CHECKING MODEL =====")
print("Model:", MODEL)

# Try Ethos-U delegate first
delegate = None
try:
    delegate = load_delegate("libethosu_delegate.so")
    print("✅ Ethos-U delegate loaded")
except Exception as e:
    print("❌ Ethos-U delegate failed:", e)
    print("Using CPU only")

# Initialize interpreter
if delegate:
    interpreter = Interpreter(model_path=MODEL,
                              experimental_delegates=[delegate])
else:
    interpreter = Interpreter(model_path=MODEL)

interpreter.allocate_tensors()

print("\n=== INPUT TENSORS ===")
inputs = interpreter.get_input_details()
for i, inp in enumerate(inputs):
    print(f"[{i}] name={inp['name']} shape={inp['shape']} quant={inp['quantization']}")

print("\n=== OUTPUT TENSORS ===")
outputs = interpreter.get_output_details()
for i, out in enumerate(outputs):
    print(f"[{i}] name={out['name']} shape={out['shape']} quant={out['quantization']}")

print("\n✅ DONE\n")

