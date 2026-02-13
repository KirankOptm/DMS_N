#!/usr/bin/env python3
"""
Test iris model eye open/close detection on laptop webcam.
Uses: face_detection_ptq.tflite + face_landmark_ptq.tflite (468 points) + iris_landmark_ptq.tflite (eye contour)
"""

import cv2
import numpy as np
import math

# Use TFLite runtime (pip install tflite-runtime) or TensorFlow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ============================================================
# Model paths (CPU versions for laptop)
# ============================================================
FACE_DETECTION_MODEL = "face_detection_ptq.tflite"
FACE_LANDMARK_MODEL = "face_landmark_ptq.tflite"
IRIS_LANDMARK_MODEL = "iris_landmark_ptq.tflite"

# BlazeFace constants
RAW_SCORE_LIMIT = 80
MIN_SUPPRESSION_THRESHOLD = 0.3

# ============================================================
# Load Face Detection Model (BlazeFace)
# ============================================================
print(f"Loading face detection model: {FACE_DETECTION_MODEL}")
face_det_interp = tflite.Interpreter(model_path=FACE_DETECTION_MODEL)
face_det_interp.allocate_tensors()
face_det_input = face_det_interp.get_input_details()[0]
face_det_outputs = face_det_interp.get_output_details()
FACE_DET_SIZE = face_det_input['shape'][1]  # Usually 128
print(f"  Input: {face_det_input['shape']}")

# Generate BlazeFace anchors
def generate_anchors():
    """Generate SSD anchors for BlazeFace (128x128 input) -> 896 anchors"""
    strides = [8, 16, 16, 16]
    anchors = []
    
    for layer_id, stride in enumerate(strides):
        feature_map_size = FACE_DET_SIZE // stride
        
        # 2 anchors per location for all layers
        num_anchors = 2
        
        for y in range(feature_map_size):
            for x in range(feature_map_size):
                for _ in range(num_anchors):
                    anchor_cx = (x + 0.5) / feature_map_size
                    anchor_cy = (y + 0.5) / feature_map_size
                    anchors.append([anchor_cx, anchor_cy])
    
    return np.array(anchors, dtype=np.float32)

ANCHORS = generate_anchors()
print(f"  Generated {len(ANCHORS)} anchors")

def decode_boxes(raw_boxes, anchors):
    """Decode BlazeFace box predictions"""
    boxes = np.zeros_like(raw_boxes[:, :4])
    
    # Center offset
    boxes[:, 0] = raw_boxes[:, 0] / FACE_DET_SIZE + anchors[:, 0]  # cx
    boxes[:, 1] = raw_boxes[:, 1] / FACE_DET_SIZE + anchors[:, 1]  # cy
    boxes[:, 2] = raw_boxes[:, 2] / FACE_DET_SIZE  # w
    boxes[:, 3] = raw_boxes[:, 3] / FACE_DET_SIZE  # h
    
    # Convert to x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    return np.stack([x1, y1, x2, y2], axis=1)

def nms(boxes, scores, threshold=0.3):
    """Non-maximum suppression"""
    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep

def detect_face(frame):
    """Detect face using BlazeFace, return (x1,y1,x2,y2) in pixels or None"""
    h, w = frame.shape[:2]
    
    # Preprocess
    img = cv2.resize(frame, (FACE_DET_SIZE, FACE_DET_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 128.0) / 128.0
    img = np.expand_dims(img, axis=0)
    
    face_det_interp.set_tensor(face_det_input['index'], img)
    face_det_interp.invoke()
    
    # Get outputs: scores [1, 896, 1], boxes [1, 896, 16]
    raw_scores = face_det_interp.get_tensor(face_det_outputs[0]['index']).squeeze()
    raw_boxes = face_det_interp.get_tensor(face_det_outputs[1]['index']).squeeze()
    
    # Sigmoid on scores (clamp to avoid overflow)
    raw_scores = np.clip(raw_scores, -RAW_SCORE_LIMIT, RAW_SCORE_LIMIT)
    scores = 1.0 / (1.0 + np.exp(-raw_scores))
    
    # Filter by threshold
    threshold = 0.5
    mask = scores > threshold
    if not np.any(mask):
        return None
    
    filtered_scores = scores[mask]
    filtered_boxes = raw_boxes[mask]
    filtered_anchors = ANCHORS[mask]
    
    # Decode boxes
    boxes = decode_boxes(filtered_boxes, filtered_anchors)
    
    # NMS
    keep = nms(boxes, filtered_scores, MIN_SUPPRESSION_THRESHOLD)
    if len(keep) == 0:
        return None
    
    # Take best detection
    best = keep[0]
    x1, y1, x2, y2 = boxes[best]
    
    # Convert to pixel coordinates
    x1 = int(np.clip(x1 * w, 0, w))
    y1 = int(np.clip(y1 * h, 0, h))
    x2 = int(np.clip(x2 * w, 0, w))
    y2 = int(np.clip(y2 * h, 0, h))
    
    return (x1, y1, x2, y2)

# ============================================================
# Load Face Landmark Model (468 points)
# ============================================================
print(f"Loading face landmark model: {FACE_LANDMARK_MODEL}")
face_lm_interp = tflite.Interpreter(model_path=FACE_LANDMARK_MODEL)
face_lm_interp.allocate_tensors()
face_lm_input = face_lm_interp.get_input_details()[0]
face_lm_outputs = face_lm_interp.get_output_details()
FACE_INPUT_SIZE = face_lm_input['shape'][1]  # Usually 192
print(f"  Input: {face_lm_input['shape']}")

# Find the output with 1404 values (468 landmarks * 3)
face_lm_output_idx = 0
for i, od in enumerate(face_lm_outputs):
    print(f"  Output {i}: {od['shape']}")
    total_size = np.prod(od['shape'])
    if total_size == 1404:
        face_lm_output_idx = i
        print(f"  -> Using output {i} for landmarks (1404 values)")

# ============================================================
# Load Iris Landmark Model (71 eye contour + 5 iris points)
# ============================================================
print(f"Loading iris landmark model: {IRIS_LANDMARK_MODEL}")
iris_interp = tflite.Interpreter(model_path=IRIS_LANDMARK_MODEL)
iris_interp.allocate_tensors()
iris_input = iris_interp.get_input_details()[0]
iris_outputs = iris_interp.get_output_details()
IRIS_INPUT_SIZE = iris_input['shape'][1]  # Usually 64
print(f"  Input: {iris_input['shape']}")
for i, od in enumerate(iris_outputs):
    print(f"  Output {i}: {od['shape']}")

# Eye landmark indices from 468 face landmarks
LEFT_EYE_START = 33
LEFT_EYE_END = 133
RIGHT_EYE_START = 362
RIGHT_EYE_END = 263
ROI_SCALE = 2

# Eye contour connections for drawing
EYE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 9), (8, 14),
]


def get_face_landmarks(face_crop_bgr):
    """Run face landmark model on face crop, return 468 normalized landmarks"""
    img = cv2.resize(face_crop_bgr, (FACE_INPUT_SIZE, FACE_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 128.0) / 128.0  # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)
    
    face_lm_interp.set_tensor(face_lm_input['index'], img)
    face_lm_interp.invoke()
    
    # Get landmarks output (1404 values = 468 * 3)
    raw = face_lm_interp.get_tensor(face_lm_outputs[face_lm_output_idx]['index'])
    landmarks = raw.reshape(-1, 3) / FACE_INPUT_SIZE  # Normalize to 0-1
    return landmarks


def get_eye_roi(landmarks, side, w, h):
    """Get square eye ROI using NXP method (ROI_SCALE=2)"""
    if side == 0:  # Left eye
        x1, y1 = landmarks[LEFT_EYE_START][:2]
        x2, y2 = landmarks[LEFT_EYE_END][:2]
    else:  # Right eye
        x1, y1 = landmarks[RIGHT_EYE_START][:2]
        x2, y2 = landmarks[RIGHT_EYE_END][:2]
    
    # Convert to pixels
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    half_w = abs(x2 - x1) // 2
    
    # Square ROI centered on eye
    roi_xmin = max(0, mid_x - ROI_SCALE * half_w)
    roi_xmax = min(w, mid_x + ROI_SCALE * half_w)
    roi_ymin = max(0, mid_y - ROI_SCALE * half_w)
    roi_ymax = min(h, mid_y + ROI_SCALE * half_w)
    
    return roi_xmin, roi_ymin, roi_xmax, roi_ymax


def get_iris_landmarks(eye_crop_bgr, side):
    """Run iris model on eye crop, return eye contour and iris landmarks"""
    if eye_crop_bgr is None or eye_crop_bgr.size == 0:
        return None, None
    
    # Right eye: flip horizontally (NXP convention)
    if side == 1:
        eye_crop_bgr = cv2.flip(eye_crop_bgr, 1)
    
    # Preprocess: resize, BGR->RGB, normalize to [-1, 1]
    img = cv2.resize(eye_crop_bgr, (IRIS_INPUT_SIZE, IRIS_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 128.0) / 128.0
    img = np.expand_dims(img, axis=0)
    
    iris_interp.set_tensor(iris_input['index'], img)
    iris_interp.invoke()
    
    # Output 0: eye contour (71 points), Output 1: iris (5 points)
    eye_contour = iris_interp.get_tensor(iris_outputs[0]['index']).reshape(-1, 3).astype(np.float32)
    iris_points = iris_interp.get_tensor(iris_outputs[1]['index']).reshape(-1, 3).astype(np.float32)
    
    # Normalize by input size
    eye_contour /= IRIS_INPUT_SIZE
    iris_points /= IRIS_INPUT_SIZE
    
    # Right eye: invert x-coordinates
    if side == 1:
        eye_contour[:, 0] = 1.0 - eye_contour[:, 0]
        iris_points[:, 0] = 1.0 - iris_points[:, 0]
    
    return eye_contour, iris_points


def blinking_ratio(eye_contour, side):
    """Calculate eye openness ratio (NXP method): height/width - NOT RELIABLE FOR BLINK"""
    if eye_contour is None or len(eye_contour) < 15:
        return 0.0
    
    # Points 0,8 = horizontal span; 4,12 = vertical span
    if side == 0:
        pt_left, pt_right = eye_contour[0], eye_contour[8]
    else:
        pt_left, pt_right = eye_contour[8], eye_contour[0]
    
    pt_top = eye_contour[12]
    pt_bottom = eye_contour[4]
    
    width = math.hypot(pt_right[0] - pt_left[0], pt_right[1] - pt_left[1])
    height = math.hypot(pt_bottom[0] - pt_top[0], pt_bottom[1] - pt_top[1])
    
    return height / width if width > 0 else 0.0


# ============================================================
# EAR (Eye Aspect Ratio) from 468 face landmarks - RELIABLE FOR BLINK
# ============================================================
# Eye landmark indices from MediaPipe 468-point face mesh
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # outer, top1, top2, inner, bottom2, bottom1
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]  # outer, top1, top2, inner, bottom2, bottom1

def get_ear(landmarks, eye_indices, crop_w, crop_h):
    """
    Calculate Eye Aspect Ratio from face landmarks.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    
    This tracks actual eyelid position, unlike iris model.
    Open eye: EAR ~0.25-0.35
    Closed eye: EAR ~0.05-0.15
    """
    def pt(i):
        return np.array([landmarks[i][0] * crop_w, landmarks[i][1] * crop_h])
    
    # Vertical distances (top to bottom of eye)
    A = np.linalg.norm(pt(eye_indices[1]) - pt(eye_indices[5]))  # p2-p6
    B = np.linalg.norm(pt(eye_indices[2]) - pt(eye_indices[4]))  # p3-p5
    
    # Horizontal distance (corner to corner)
    C = np.linalg.norm(pt(eye_indices[0]) - pt(eye_indices[3]))  # p1-p4
    
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear


def draw_landmarks(frame, landmarks, face_box):
    """Draw 468 face landmarks on frame"""
    fx1, fy1, fw, fh = face_box
    for i, (x, y, z) in enumerate(landmarks):
        px = int(fx1 + x * fw)
        py = int(fy1 + y * fh)
        cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)


def draw_eye_contour(frame, eye_contour, roi):
    """Draw eye contour (71 points) on frame"""
    if eye_contour is None:
        return
    
    rx1, ry1, rx2, ry2 = roi
    rw, rh = rx2 - rx1, ry2 - ry1
    
    # Draw connections
    for (i1, i2) in EYE_CONNECTIONS:
        p1 = (int(rx1 + eye_contour[i1][0] * rw), int(ry1 + eye_contour[i1][1] * rh))
        p2 = (int(rx1 + eye_contour[i2][0] * rw), int(ry1 + eye_contour[i2][1] * rh))
        cv2.line(frame, p1, p2, (255, 0, 0), 1)


def draw_iris(frame, iris_points, roi):
    """Draw iris center (5 points) on frame"""
    if iris_points is None:
        return
    
    rx1, ry1, rx2, ry2 = roi
    rw, rh = rx2 - rx1, ry2 - ry1
    
    for (x, y, z) in iris_points:
        px = int(rx1 + x * rw)
        py = int(ry1 + y * rh)
        cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)


# ============================================================
# Main loop
# ============================================================
print("\nStarting webcam... Press 'q' to quit")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Track previous EAR state to only print on change
prev_ear_state = None
smoothed_ear = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    
    # Detect face using BlazeFace
    face_box = detect_face(frame)
    
    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        fw, fh = fx2 - fx1, fy2 - fy1
        
        # Draw face box
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
        
        # Crop face with margin
        margin = 0.2
        mx, my = int(fw * margin), int(fh * margin)
        cx1 = max(0, fx1 - mx)
        cy1 = max(0, fy1 - my)
        cx2 = min(w, fx2 + mx)
        cy2 = min(h, fy2 + my)
        face_crop = frame[cy1:cy2, cx1:cx2]
        
        if face_crop.size > 0:
            # Get 468 face landmarks
            landmarks = get_face_landmarks(face_crop)
            crop_w = cx2 - cx1
            crop_h = cy2 - cy1
            
            # Draw all 468 landmarks
            for i, (lx, ly, lz) in enumerate(landmarks):
                px = int(cx1 + lx * crop_w)
                py = int(cy1 + ly * crop_h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
            
            # ============================================================
            # EAR-based eye open/close detection (from 468 face landmarks)
            # Reference: dms_integrated_mdp_yolo_log.py approach
            # ============================================================
            left_ear = get_ear(landmarks, LEFT_EYE_EAR, crop_w, crop_h)
            right_ear = get_ear(landmarks, RIGHT_EYE_EAR, crop_w, crop_h)
            raw_avg_ear = (left_ear + right_ear) / 2.0
            
            # Adaptive smoothing (MediaPipe strategy: fast close, slow open)
            if smoothed_ear is None:
                smoothed_ear = raw_avg_ear
            else:
                if raw_avg_ear < smoothed_ear:  # Eyes closing
                    SMOOTH_FACTOR = 0.7  # Very fast response for closing
                else:  # Eyes opening
                    SMOOTH_FACTOR = 0.25  # Slower for opening (reduce jitter)
                smoothed_ear = SMOOTH_FACTOR * raw_avg_ear + (1 - SMOOTH_FACTOR) * smoothed_ear
            
            avg_ear = smoothed_ear
            
            # Eye openness percentage (calibrated for TFLite model + 45-degree camera)
            # Adjusted range so 100% is reachable: your OPEN ~0.26+, CLOSED ~0.18
            EAR_FULLY_OPEN = 0.26
            EAR_FULLY_CLOSED = 0.18
            
            if avg_ear <= EAR_FULLY_CLOSED:
                eye_openness = 0.0
            elif avg_ear >= EAR_FULLY_OPEN:
                eye_openness = 100.0
            else:
                eye_openness = ((avg_ear - EAR_FULLY_CLOSED) / (EAR_FULLY_OPEN - EAR_FULLY_CLOSED)) * 100.0
            
            # 3 states: CLOSED (0-33%), MILD (33-66%), OPEN (66-100%)
            if eye_openness <= 33.0:
                state = "CLOSED"
                color = (0, 0, 255)  # Red
            elif eye_openness <= 66.0:
                state = "MILD"
                color = (0, 255, 255)  # Yellow
            else:
                state = "OPEN"
                color = (0, 255, 0)  # Green
            
            # Display on frame (stationary position)
            cv2.putText(frame, f"Eye: {state} ({eye_openness:.0f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.3f} (L:{left_ear:.2f} R:{right_ear:.2f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Stationary console output (overwrite same line)
            print(f"\r[Eye] {eye_openness:5.1f}% {state:6s} EAR={avg_ear:.3f}  ", end="", flush=True)
            
            # Only print newline when state changes
            if state != prev_ear_state:
                print(f"\n[Eye] State changed: {prev_ear_state} -> {state}")
                prev_ear_state = state
            
            # ============================================================
            # Still draw iris for visualization (but NOT for blink detection)
            # ============================================================
            for side, eye_name in [(0, "Left"), (1, "Right")]:
                # Get eye ROI (square)
                roi = get_eye_roi(landmarks, side, crop_w, crop_h)
                ex1, ey1, ex2, ey2 = roi
                
                # Adjust to frame coordinates
                ex1_f = cx1 + ex1
                ey1_f = cy1 + ey1
                ex2_f = cx1 + ex2
                ey2_f = cy1 + ey2
                
                # Clip to frame bounds
                ex1_f = max(0, min(w, ex1_f))
                ey1_f = max(0, min(h, ey1_f))
                ex2_f = max(0, min(w, ex2_f))
                ey2_f = max(0, min(h, ey2_f))
                
                if ex2_f <= ex1_f or ey2_f <= ey1_f:
                    continue
                
                eye_crop = frame[ey1_f:ey2_f, ex1_f:ex2_f]
                
                if eye_crop.size == 0:
                    continue
                
                # Draw eye ROI box
                cv2.rectangle(frame, (ex1_f, ey1_f), (ex2_f, ey2_f), (0, 0, 255), 1)
                
                # Get iris/eye landmarks (for visualization only)
                eye_contour, iris_pts = get_iris_landmarks(eye_crop, side)
                
                # Draw eye contour and iris
                draw_eye_contour(frame, eye_contour, (ex1_f, ey1_f, ex2_f, ey2_f))
                draw_iris(frame, iris_pts, (ex1_f, ey1_f, ex2_f, ey2_f))
    
    cv2.imshow("Iris Eye Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
