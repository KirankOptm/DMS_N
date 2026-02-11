import cv2
import numpy as np
import os

# Disable XNNPACK delegate to avoid compatibility issues with int8 models
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "eye_openness_int8.tflite"
IMG_SIZE = 96

# ===============================
# LOAD TFLITE MODEL
# ===============================
interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale, in_zero = input_details[0]["quantization"]
out_scale, out_zero = output_details[0]["quantization"]

print("[INFO] Model loaded")
print("Input scale:", in_scale, "zero:", in_zero)
print("Output scale:", out_scale, "zero:", out_zero)

# ===============================
# PREPROCESS FUNCTION
# ===============================
def preprocess_eye(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Direct int8 conversion (NO float normalization)
    img = img.astype(np.int8) - 128

    return np.expand_dims(img, axis=0)

# ===============================
# CAMERA TEST
# ===============================
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Camera not accessible"

print("[INFO] Camera opened. Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- TEMP ROI (center crop for test) ----
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    size = min(w, h) // 3
    eye_roi = frame[
        cy - size//2 : cy + size//2,
        cx - size//2 : cx + size//2
    ]

    if eye_roi.size == 0:
        continue

    input_tensor = preprocess_eye(eye_roi)

    # ---- INFERENCE ----
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(output_details[0]["index"])

    # ---- DEQUANTIZE ----
    output_int8_val = output_int8[0][0]
    prob = (output_int8_val + 128) * 0.003906

    # Production-safe decision logic
    if prob > 0.55:
        state = "OPEN"
    elif prob < 0.45:
        state = "CLOSED"
    else:
        state = "UNCERTAIN"

    # ---- VISUALIZE ----
    cv2.rectangle(
        frame,
        (cx - size//2, cy - size//2),
        (cx + size//2, cy + size//2),
        (0, 255, 0), 2
    )
    cv2.putText(
        frame,
        f"{state} ({prob:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Eye Openness INT8 Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
