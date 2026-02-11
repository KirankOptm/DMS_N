"""
PMU Profiler for individual NPU models.
Runs each model separately with PMU counters enabled to measure:
  - Cycle count (total NPU cycles)
  - AXI0 reads (internal SRAM - fast path)
  - AXI1 reads (external DDR - slow path)
  - Active NPU cycles (computing)
  - Idle NPU cycles (waiting for data)

Usage on board:
  python3 test_pmu_models.py
  python3 test_pmu_models.py --runs 20
  python3 test_pmu_models.py --model face_detection_ptq_vela.tflite
"""

import cv2
import time
import argparse
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    print("[Runtime] Using tflite_runtime")
except ImportError:
    import tensorflow.lite as tflite
    print("[Runtime] Using tensorflow.lite")


def load_model_with_pmu(model_path):
    """Load a TFLite model with PMU counters enabled"""
    try:
        delegate = tflite.load_delegate(
            "/usr/lib/libethosu_delegate.so",
            {
                "device_name": "/dev/ethosu0",
                "cache_file_path": ".",
                "enable_cycle_counter": "1",
                "pmu_event0": "3",     # AXI0 reads (internal SRAM)
                "pmu_event1": "4",     # AXI1 reads (external DDR)
                "pmu_event2": "1",     # Active NPU cycles
                "pmu_event3": "2",     # Idle NPU cycles
            }
        )
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate]
        )
        print(f"  [NPU] Loaded with Ethos-U delegate + PMU")
    except Exception as e:
        print(f"  [ERROR] Delegate failed: {e}")
        return None

    interpreter.allocate_tensors()
    return interpreter


def get_camera_frame():
    """Capture a single frame from camera"""
    import glob
    video_devices = sorted(glob.glob('/dev/video*'))
    device_num = 0
    if video_devices:
        device_num = int(video_devices[0].split('video')[-1])

    cap = cv2.VideoCapture(device_num)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera, using synthetic frame")
        cap.release()
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Read a few frames to let camera stabilize
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Failed to read frame, using synthetic")
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print(f"  [Camera] Captured frame: {frame.shape[1]}x{frame.shape[0]}")
    return frame


def preprocess_for_model(frame_bgr, input_shape):
    """Preprocess frame for any model: resize, BGR->RGB, normalize to [-1,1]"""
    h, w = input_shape[1], input_shape[2]
    img = cv2.resize(frame_bgr, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 128.0) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


def run_pmu_test(model_path, model_name, frame_bgr, num_runs=10):
    """Run a single model multiple times and collect PMU stats"""
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"  File:  {model_path}")
    print(f"  Runs:  {num_runs}")
    print(f"{'='*60}")

    interpreter = load_model_with_pmu(model_path)
    if interpreter is None:
        return

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    input_shape = input_details['shape']

    print(f"  Input:  {input_shape} {input_details['dtype']}")
    for i, od in enumerate(output_details):
        print(f"  Output {i}: {od['shape']} {od['dtype']}")

    # Prepare input
    inp = preprocess_for_model(frame_bgr, input_shape)

    # Warmup run (first run is always slower due to caching)
    interpreter.set_tensor(input_details['index'], inp)
    interpreter.invoke()
    print(f"\n  [Warmup done]\n")

    # Collect PMU data over multiple runs
    wall_times = []

    print(f"  {'Run':>4} | {'Wall(ms)':>9} | {'Cycles':>12} | {'AXI0(SRAM)':>12} | {'AXI1(DDR)':>12} | {'Active':>12} | {'Idle':>12} | {'Util%':>6}")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}")

    cycles_list = []
    axi0_list = []
    axi1_list = []
    active_list = []
    idle_list = []

    for run in range(num_runs):
        interpreter.set_tensor(input_details['index'], inp)

        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1000
        wall_times.append(wall_ms)

        # PMU values are printed to stdout by the delegate
        # We also measure wall time for comparison
        print(f"  {run+1:>4} | {wall_ms:>9.2f} |")

    # Summary
    print(f"\n  --- WALL TIME SUMMARY ({model_name}) ---")
    print(f"  Min:    {min(wall_times):.2f} ms")
    print(f"  Max:    {max(wall_times):.2f} ms")
    print(f"  Mean:   {np.mean(wall_times):.2f} ms")
    print(f"  Median: {np.median(wall_times):.2f} ms")
    print(f"  Std:    {np.std(wall_times):.2f} ms")
    print()


def main():
    parser = argparse.ArgumentParser(description="PMU Profiler for NPU models")
    parser.add_argument('--runs', type=int, default=10, help="Number of inference runs per model")
    parser.add_argument('--model', type=str, default=None, help="Test a specific model only")
    parser.add_argument('--face_detection', type=str, default='face_detection_ptq_vela.tflite')
    parser.add_argument('--face_landmark', type=str, default='face_landmark_ptq_vela.tflite')
    parser.add_argument('--iris_landmark', type=str, default='iris_landmark_ptq_vela.tflite')
    args = parser.parse_args()

    print("=" * 60)
    print("  NPU PMU PROFILER — Ethos-U65")
    print("  PMU Events:")
    print("    event0 = 3 (AXI0 reads / internal SRAM)")
    print("    event1 = 4 (AXI1 reads / external DDR)")
    print("    event2 = 1 (Active NPU cycles)")
    print("    event3 = 2 (Idle NPU cycles)")
    print("=" * 60)

    # Capture a real camera frame
    print("\n[Step 1] Capturing camera frame...")
    frame = get_camera_frame()

    # Also prepare an eye crop (for iris model testing)
    # Use center 64x64 region as a rough eye crop proxy
    h, w = frame.shape[:2]
    eye_crop = frame[h//3:h//3+100, w//3:w//3+100]

    models = []
    if args.model:
        models.append((args.model, args.model, frame))
    else:
        models = [
            (args.face_detection, "BlazeFace (Face Detection)", frame),
            (args.face_landmark, "Face Landmark (468 points)", frame),
            (args.iris_landmark, "Iris Landmark (eye contour + iris)", eye_crop),
        ]

    print(f"\n[Step 2] Running PMU tests ({args.runs} runs each)...")

    total_mean = 0
    for model_path, model_name, input_frame in models:
        run_pmu_test(model_path, model_name, input_frame, num_runs=args.runs)

    print("\n" + "=" * 60)
    print("  PMU PROFILING COMPLETE")
    print("  NOTE: PMU counter values are printed by the Ethos-U delegate")
    print("  above each 'Run' line. Look for lines like:")
    print("    cycle_counter : XXXXXXX")
    print("    pmu_event_0   : XXXXXXX  (AXI0/SRAM)")
    print("    pmu_event_1   : XXXXXXX  (AXI1/DDR)")
    print("    pmu_event_2   : XXXXXXX  (Active)")
    print("    pmu_event_3   : XXXXXXX  (Idle)")
    print("=" * 60)
    print("\n  Key metrics to compare:")
    print("  - Active / (Active + Idle) = NPU utilization %")
    print("  - AXI1 >> AXI0 = memory-bound (DDR bottleneck)")
    print("  - AXI0 >> AXI1 = compute-bound (good, using SRAM)")
    print("  - cycle_counter = total NPU cycles per inference")


if __name__ == "__main__":
    main()
