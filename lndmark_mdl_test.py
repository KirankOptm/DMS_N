import cv2
import time
import argparse
import numpy as np
from collections import deque
from datetime import datetime
from collections import defaultdict
import glob
import os
import tensorflow as tf
from scrfd_detector import SCRFDDetector


# ============================================================
# PFLD 98-point Landmark Detector
# ============================================================

class PFLDLandmarkDetector:
    """PFLD 98-point Landmark Detector - 112x112 input with temporal smoothing"""
    def __init__(self, model_path="pfld_int8.tflite"):
        print(f"[PFLD-98] Loading model: {model_path}")
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        self.input_scale, self.input_zero = self.input_details['quantization']
        self.output_scale, self.output_zero = self.output_details['quantization']
        
        self.input_shape = self.input_details['shape']
        self.input_size = self.input_shape[1]
        
        self.prev_landmarks = None
        self.smoothing_alpha = 0.8
        self.frame_count = 0
        
        print(f"[PFLD-98] Model loaded successfully")
    
    def detect(self, frame, face_box, scrfd_keypoints=None):
        if scrfd_keypoints is None:
            return None
        
        keypoints = np.array(scrfd_keypoints).reshape(5, 2)
        left_eye = keypoints[0]
        right_eye = keypoints[1]
        
        target_left_eye = np.array([34, 34])
        target_right_eye = np.array([78, 34])
        
        src_pts = np.array([left_eye, right_eye], dtype=np.float32)
        dst_pts = np.array([target_left_eye, target_right_eye], dtype=np.float32)
        
        transform_matrix = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        aligned_face = cv2.warpAffine(frame, transform_matrix, (112, 112), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        input_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face_normalized = input_rgb.astype(np.float32) / 255.0
        face_quantized = (face_normalized / self.input_scale) + self.input_zero
        face_quantized = np.clip(face_quantized, -128, 127).astype(np.int8)
        face_input = np.expand_dims(face_quantized, axis=0)
        
        self.interpreter.set_tensor(self.input_details['index'], face_input)
        self.interpreter.invoke()
        
        landmarks_int8 = self.interpreter.get_tensor(self.output_details['index'])
        landmarks_float = (landmarks_int8.astype(np.float32) - self.output_zero) * self.output_scale
        landmarks_aligned = landmarks_float.reshape(-1, 2) * 112.0
        
        A = transform_matrix[:, :2]
        b = transform_matrix[:, 2]
        A_inv = np.linalg.inv(A)
        
        landmarks_frame = np.zeros_like(landmarks_aligned)
        for i in range(len(landmarks_aligned)):
            pt_aligned = landmarks_aligned[i] - b
            landmarks_frame[i] = A_inv @ pt_aligned
        
        self.frame_count += 1
        if self.prev_landmarks is not None and self.frame_count > 3:
            landmarks_frame = self.smoothing_alpha * landmarks_frame + (1 - self.smoothing_alpha) * self.prev_landmarks
        
        self.prev_landmarks = landmarks_frame.copy()
        return landmarks_frame.astype(np.float32)


# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--ear_threshold', type=float, default=0.180)
parser.add_argument('--eye_closed_frames_threshold', type=int, default=9)
parser.add_argument('--blink_rate_threshold', type=int, default=5)
parser.add_argument('--mar_threshold', type=float, default=0.75)
parser.add_argument('--yawn_threshold', type=int, default=3)
parser.add_argument('--frame_width', type=int, default=1280)
parser.add_argument('--frame_height', type=int, default=720)
parser.add_argument('--gaze_deviation_threshold', type=float, default=0.05, help="Threshold for gaze deviation from center (0 to 1)")
parser.add_argument('--scale_factor', type=float, default=1.0, help="Scaling factor for resolution (e.g., 0.5x, 1x, 2x)")
parser.add_argument('--head_turn_threshold', type=float, default=0.08, help="Threshold for head turning detection")
parser.add_argument('--calibration_time', type=int, default=15, help="Calibration duration in seconds")
parser.add_argument('--no_mesh_display', action='store_true', help="Disable landmark drawing on frame")
parser.add_argument('--camera_device', type=int, default=0, help="Camera device index (0, 1, 2, etc.)")

args = parser.parse_args()

# Setup SCRFD + PFLD
print("[INFO] Initializing SCRFD + PFLD system...")
scrfd_detector = SCRFDDetector("scrfd_500m_full_int8.tflite")
pfld_detector = PFLDLandmarkDetector("pfld_int8.tflite")

# 98-point PFLD landmark indices
LEFT_EYE_98 = list(range(60, 68))    # Points 60-67 (8 points)
RIGHT_EYE_98 = list(range(68, 76))   # Points 68-75 (8 points)
MOUTH_OUTER_98 = list(range(76, 88)) # Points 76-87 (12 points)
NOSE_TIP_98 = 54  # Nose tip for head pose
CHIN_98 = 16  # Chin point

# Open webcam
print(f"[Camera] Opening camera {args.camera_device}...")
cap = cv2.VideoCapture(args.camera_device)

if not cap.isOpened():
    print(f"[ERROR] Cannot open camera {args.camera_device}")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.frame_width * args.scale_factor))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.frame_height * args.scale_factor))
print("[Camera] Opened successfully")

eye_closure_counter = 0
blink_counter = 0
blink_timer = time.time()
yawn_counter = 0
mar_deque = deque(maxlen=30)

ALERT_DURATION = 3
#active_alerts = {}
active_alerts = defaultdict(lambda: [None, 0])
calibration_mode = True
calibration_start_time = time.time()
calibration_duration = args.calibration_time # seconds
gaze_center = 0.5
head_center_x = 0.5
head_center_y = 0.5

print(f"[Calibration] Starting {args.calibration_time}s calibration period...")
print(f"[Calibration] Look straight ahead and keep face visible")

def add_alert(frame,message):
    ts = datetime.now().strftime("%H:%M:%S")
    active_alerts[f"{ts} {message}"] = time.time()
    print(f"[ALERT {ts}] {message}")  # Print alerts to console
    return message

def get_aspect_ratio_98(landmarks, eye_indices):
    """Calculate EAR for 98-point landmarks (already in pixel coordinates)"""
    eye = landmarks[eye_indices]
    
    # Vertical distances
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    
    # Horizontal distance
    h = np.linalg.norm(eye[0] - eye[3])
    
    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear

def get_mar_98(landmarks, mouth_indices):
    """Calculate MAR for 98-point landmarks"""
    mouth = landmarks[mouth_indices]
    
    # Vertical distances
    v1 = np.linalg.norm(mouth[2] - mouth[10])
    v2 = np.linalg.norm(mouth[4] - mouth[8])
    
    # Horizontal distance
    h = np.linalg.norm(mouth[0] - mouth[6])
    
    mar = (v1 + v2) / (2.0 * h + 1e-6)
    return mar

def draw_landmarks_98(frame, landmarks):
    """Draw 98 landmarks as green dots"""
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    

eye_closed = 0
head_turn = 0
hands_free = False
head_tilt = 0
head_droop = 0
yawn = False
msg = ""
last_msg = "Normal and Active Driving"
fid = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to read frame")
        break

    fid += 1
    h, w = frame.shape[:2]
    
    # Log frame info on first frame
    if fid == 1:
        print(f"[Camera] Frame resolution: {w}x{h}")
        print("[DMS] Processing started...")

    display_frame = frame.copy()
    
    # Detect faces with SCRFD
    faces = scrfd_detector.detect(frame, score_threshold=0.45)
    
    current_time = time.time()
    msg = last_msg
    msg = "Normal and Active Driving"
    eye_closed = 0
    head_turn = 0
    head_tilt = 0
    head_droop = 0
    yawn = False
    
    if faces:
        face = faces[0]
        box = face['box']
        score = face['score']
        keypoints = face['keypoints']
        
        x1, y1, x2, y2 = map(int, box)
        
        # Draw face box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Face: {score:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Detect 98-point landmarks with PFLD
        landmarks = pfld_detector.detect(frame, box, scrfd_keypoints=keypoints)
        
        if landmarks is not None:
            # Landmarks detected - no display for production
            
            face_center = landmarks[NOSE_TIP_98]
            
            # EYE CLOSURE - Calculate EAR
            left_ear = get_aspect_ratio_98(landmarks, LEFT_EYE_98)
            right_ear = get_aspect_ratio_98(landmarks, RIGHT_EYE_98)
            avg_ear = (left_ear + right_ear) / 2
            
            if avg_ear < args.ear_threshold:
                eye_closure_counter += 1
                
                if eye_closure_counter > 30:
                    msg = "Alert: Eyes Closed Too Long"
                    msg = add_alert(frame, msg)
                    last_msg = msg
                    eye_closed = 2
                elif eye_closure_counter > args.eye_closed_frames_threshold:
                    msg = "Warning: Eyes Closed"
                    msg = add_alert(frame, msg)
                    last_msg = msg
                    eye_closed = 1
            else:
                if 2 <= eye_closure_counter < args.eye_closed_frames_threshold:
                    blink_counter += 1
                eye_closure_counter = 0
            
            if current_time - blink_timer > 60:
                if blink_counter >= args.blink_rate_threshold:
                    msg = "High Blinking Rate"
                    msg = add_alert(frame, msg)
                    last_msg = msg
                blink_counter = 0
                blink_timer = current_time
            
            # YAWN DETECTION - Calculate MAR
            mar = get_mar_98(landmarks, MOUTH_OUTER_98)
            mar_deque.append(mar)
            
            # Debug MAR value
            if fid % 30 == 0:
                print(f"[Debug] MAR={mar:.3f}, threshold={args.mar_threshold}")
            
            if mar > args.mar_threshold:
                yawn_counter += 1
            
            if yawn_counter > args.yawn_threshold:
                msg = "Warning: Yawning"
                msg = add_alert(frame, msg)
                last_msg = msg
                yawn = True
                yawn_counter = 0
            
            # HEAD POSE - using nose tip
            head_x = face_center[0] / w
            head_y = face_center[1] / h
            
            # Debug output every 30 frames
            if fid % 30 == 0:
                print(f"[Debug] head_x={head_x:.3f}, head_y={head_y:.3f}, center_x={head_center_x:.3f}, center_y={head_center_y:.3f}")
            
            if calibration_mode:
                cv2.putText(frame, "Please align your face naturally, facing forward, before the countdown ends", (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                head_x_offset = abs(head_x - head_center_x)
                head_y_offset = abs(head_y - head_center_y)
                head_x_direction = head_x - head_center_x  # Positive = right, Negative = left
                
                # Debug head turn direction
                if fid % 30 == 0:
                    direction = "RIGHT" if head_x_direction > 0 else "LEFT"
                    print(f"[Debug] x_offset={head_x_offset:.3f}, y_offset={head_y_offset:.3f}, direction={direction}")
                
                # Head turn detection (left/right) - Production thresholds
                if head_x_offset > 0.15:
                    if head_x_offset >= 0.25:
                        msg = "Severe Head Turn"
                        msg = add_alert(frame, msg)
                        last_msg = msg
                        head_turn = 3
                    elif head_x_offset >= 0.18:
                        msg = "Moderate Head Turn"
                        msg = add_alert(frame, msg)
                        last_msg = msg
                        head_turn = 2
                    else:
                        msg = "Mild Head Turn"
                        msg = add_alert(frame, msg)
                        last_msg = msg
                        head_turn = 1
                
                # Detect Upward vs Downward head tilt - Production thresholds
                if abs(head_y_offset) > 0.12:
                    if head_y < head_center_y:
                        # Head tilted upward
                        if abs(head_y_offset) >= 0.20:
                            msg = "Severe Looking Upward"
                            msg = add_alert(frame, msg)
                            last_msg = msg
                            head_tilt = 3
                        elif abs(head_y_offset) >= 0.15:
                            msg = "Moderate Looking Upward"
                            msg = add_alert(frame, msg)
                            last_msg = msg
                            head_tilt = 2
                        else:
                            msg = "Mild Looking Upward"
                            last_msg = msg
                            head_tilt = 1
                    else:
                        # Head tilted downward
                        if abs(head_y_offset) >= 0.20:
                            msg = "Head drooped"
                            msg = add_alert(frame, msg)
                            last_msg = msg
                            head_droop = 3
                        elif abs(head_y_offset) >= 0.15:
                            msg = "Head drooping started"
                            msg = add_alert(frame, msg)
                            last_msg = msg
                            head_droop = 2
                        else:
                            msg = "Head drooping symptom"
                            msg = add_alert(frame, msg)
                            last_msg = msg
                            head_droop = 1
    
    # Combined drowsiness alerts
    if eye_closed == 2 and head_droop >= 1 or eye_closed == 2 and yawn:
        msg = "Severe DROWSINESS Observed"
        msg = add_alert(frame, msg)
        last_msg = msg
    elif eye_closed == 1 and head_droop >= 1 or eye_closed == 1 and yawn:
        msg = "Moderate DROWSINESS Observed"
        msg = add_alert(frame, msg)
        last_msg = msg
    
    # Expire alerts
    expired = [k for k, t in active_alerts.items() if current_time - t > ALERT_DURATION]
    for k in expired:
        del active_alerts[k]
    
    # Draw alerts
    for i, msg in enumerate(active_alerts):
        if "Mild" in msg or "Warning" in msg:
            color = (255, 255, 255)  # White
        elif "Moderate" in msg or "Alert" in msg:
            color = (0, 255, 255)    # Yellow
        elif "Severe" in msg:
            color = (0, 0, 255)      # Red
        else:
            color = (0, 0, 255)
        cv2.putText(frame, msg, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display metrics
    if faces and landmarks is not None:
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, h-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, h-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Calibration countdown
    if calibration_mode:
        countdown = int(calibration_duration - (time.time() - calibration_start_time))
        if countdown > 0:
            if countdown % 5 == 0 and fid % 30 == 0:
                if faces:
                    print(f"[Calibration] {countdown}s remaining - Face detected OK")
                else:
                    print(f"[Calibration] {countdown}s remaining - WARNING: No face detected!")
            cv2.putText(frame, f"Calibrating in {countdown}s...", (20, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if countdown == 0 and faces and landmarks is not None:
            head_center_x = head_x
            head_center_y = head_y
            calibration_mode = False
            print(f"[Calibration] COMPLETE - Head center: ({head_center_x:.3f}, {head_center_y:.3f})")
            add_alert(frame, "Calibration Complete")
        elif countdown == 0 and not faces:
            calibration_mode = False
            print(f"[Calibration] WARNING - Completed without face detection! Using defaults.")
    
    cv2.imshow("DMS - PFLD 98-Point Landmarks", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

print("\n[DMS] Shutting down...")
cap.release()
cv2.destroyAllWindows()

