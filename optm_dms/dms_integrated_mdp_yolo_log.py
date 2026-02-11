import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8')

import argparse
import os
import time
import math
from collections import deque, defaultdict
import socket
import struct

import cv2
import numpy as np

# Latency Tracker for pipeline analysis
try:
    import dms_latency_tracker as lat
    _LAT = True
except ImportError:
    _LAT = False

# Custom print wrapper with [OPTM] prefix
from datetime import datetime
def optm_print(prefix, message):
    """Print with [OPTM] [timestamp] prefix"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[OPTM] [{timestamp}] [{prefix}] {message}")

# Gaze Tracker for eye direction detection
try:
    from gaze_tracker import GazeTracker
    _HAS_GAZE_TRACKER = True
except ImportError:
    _HAS_GAZE_TRACKER = False

# KSS Calculator for AIS 184 Compliance
from kss_calculator import create_kss_system



# Buzzer Control (Hardware LED/Buzzer via sysfs)
BUZZER = "/sys/class/leds/green:user1/brightness"
_BUZZER_AVAILABLE = None

def _check_buzzer():
    """Check if buzzer hardware is available"""
    global _BUZZER_AVAILABLE
    if _BUZZER_AVAILABLE is None:
        try:
            with open(BUZZER, "w") as f:
                f.write("0")
            _BUZZER_AVAILABLE = True
        except Exception as e:
            _BUZZER_AVAILABLE = False
    return _BUZZER_AVAILABLE

def buzzer_on():
    """Turn buzzer ON"""
    if not _check_buzzer():
        return
    try:
        with open(BUZZER, "w") as f:
            f.write("1")
    except Exception as e:
        print(f"[Buzzer] Error turning ON: {e}")

def buzzer_off():
    """Turn buzzer OFF"""
    if not _check_buzzer():
        return
    try:
        with open(BUZZER, "w") as f:
            f.write("0")
    except Exception as e:
        print(f"[Buzzer] Error turning OFF: {e}")

def buzzer_beep(times=3, on_s=0.07, off_s=0.07):
    """Beep buzzer N times with specified duration"""
    if not _check_buzzer():
        return
    optm_print("Buzzer", f"Beeping {times} time(s)...")
    for _ in range(times):
        buzzer_on()
        time.sleep(on_s)
        buzzer_off()
        time.sleep(off_s)

# Face Authentication Module
try:
    from authenticate_face_board import quick_authenticate
    _HAS_AUTH = True
except ImportError:
    _HAS_AUTH = False

# Inlined dms_mt utilities (camera, mediapipe_utils, yolo_utils)
# Camera open helper - matches DMSv8
import sys as _sys
import glob as _glob

def find_available_camera():
    """Auto-detect available camera - DMSv8 style"""
    video_devices = sorted(_glob.glob('/dev/video*'))
    if video_devices:
        for device in video_devices:
            try:
                device_num = int(device.split('video')[-1])
                test_cap = cv2.VideoCapture(device_num)
                if test_cap.isOpened():
                    test_cap.release()
                    return device_num
                test_cap.release()
            except:
                continue
    
    for i in range(6):
        try:
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                test_cap.release()
                return i
            test_cap.release()
        except:
            continue
    
    return None

# FrameSender: TCP client to send frames to remote server
class FrameSender:
    """Send frames to remote server via TCP"""
    
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connected = False
        self.frame_count = 0
        
    def connect(self):
        """Connect to remote server with 3-second timeout"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(3.0)  # 3-second timeout (was blocking forever)
            self.socket.connect((self.server_ip, self.server_port))
            self.socket.settimeout(None)  # Reset to blocking after connection
            self.connected = True
            print(f"[FrameSender] ✓ Connected to {self.server_ip}:{self.server_port}")
            return True
        except socket.timeout:
            print(f"[FrameSender] ✗ Connection timeout (3s) to {self.server_ip}:{self.server_port}")
            self.connected = False
            return False
        except Exception as e:
            print(f"[FrameSender] ✗ Failed to connect: {e}")
            self.connected = False
            return False
    
    def send_frame(self, frame):
        """Send frame to remote server"""
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                return False
            
            frame_data = jpeg.tobytes()
            frame_size = len(frame_data)
            
            # Send frame size (4 bytes) + frame data
            self.socket.sendall(struct.pack('!I', frame_size))
            self.socket.sendall(frame_data)
            
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                print(f"[FrameSender] Sent {self.frame_count} frames")
            
            return True
            
        except Exception as e:
            print(f"[FrameSender] Send error: {e}")
            self.connected = False
            if self.socket:
                self.socket.close()
            return False
    
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("[FrameSender] Connection closed")

# MediaPipe-related helpers and constants
import math as _math
try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    mp = None
    _HAS_MP = False

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EAR_TIP = 234
RIGHT_EAR_TIP = 454
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

def create_solutions():
    if not _HAS_MP:
        return None, None, None, None, None, None
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    hand_connections = mp_hands.HAND_CONNECTIONS
    face_mesh_connections = mp_face_mesh.FACEMESH_TESSELATION
    return (face_mesh, hands, pose, mp_drawing, hand_connections, face_mesh_connections)

def get_aspect_ratio(landmarks, eye_indices, w, h):
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])
    A = np.linalg.norm(pt(eye_indices[1]) - pt(eye_indices[5]))
    B = np.linalg.norm(pt(eye_indices[2]) - pt(eye_indices[4]))
    C = np.linalg.norm(pt(eye_indices[0]) - pt(eye_indices[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0

def get_mar(landmarks, mouth_idx, w, h):
    top = np.array([landmarks[mouth_idx[0]].x * w, landmarks[mouth_idx[0]].y * h])
    bottom = np.array([landmarks[mouth_idx[1]].x * w, landmarks[mouth_idx[1]].y * h])
    left = np.array([landmarks[mouth_idx[2]].x * w, landmarks[mouth_idx[2]].y * h])
    right = np.array([landmarks[mouth_idx[3]].x * w, landmarks[mouth_idx[3]].y * h])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal if horizontal > 0 else 0

def get_iris_center_and_radius(landmarks, indices, w, h):
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float64)
    center = points.mean(axis=0)
    if len(points) > 0:
        dists = np.sqrt(((points - center) ** 2).sum(axis=1))
        radius = float(dists.mean()) if dists.size > 0 else 1.0
    else:
        radius = 1.0
    return (int(center[0]), int(center[1])), max(1.0, radius)

def _build_camera_intrinsics(w, h):
    focal_length = float(max(w, h))
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs

def _get_face_image_points(landmarks, w, h):
    try:
        return np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[226].x * w, landmarks[226].y * h),
            (landmarks[446].x * w, landmarks[446].y * h),
            (landmarks[57].x * w, landmarks[57].y * h),
            (landmarks[287].x * w, landmarks[287].y * h)
        ], dtype=np.float64)
    except Exception:
        return None

def estimate_gaze_direction(landmarks, w, h):
    camera_matrix, dist_coeffs = _build_camera_intrinsics(w, h)
    image_points = _get_face_image_points(landmarks, w, h)
    if image_points is None:
        return None, None
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    nose_end_3d = np.array([(0.0, 0.0, 1000.0)], dtype=np.float64)
    nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs)
    nose_tip = (int(image_points[0][0]), int(image_points[0][1]))
    nose_proj = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
    vx, vy = float(nose_proj[0] - nose_tip[0]), float(nose_proj[1] - nose_tip[1])
    mag = _math.hypot(vx, vy)
    if mag > 0:
        vx, vy = vx / mag, vy / mag
    return (vx, vy), nose_tip

def compute_head_angles(landmarks, w, h):
    camera_matrix, dist_coeffs = _build_camera_intrinsics(w, h)
    image_points = _get_face_image_points(landmarks, w, h)
    if image_points is None:
        return None
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6
    if not singular:
        x = _math.degrees(_math.atan2(R[2, 1], R[2, 2]))
        y = _math.degrees(_math.atan2(-R[2, 0], sy))
        z = _math.degrees(_math.atan2(R[1, 0], R[0, 0]))
    else:
        x = _math.degrees(_math.atan2(-R[1, 2], R[1, 1]))
        y = _math.degrees(_math.atan2(-R[2, 0], sy))
        z = 0.0
    return (y, x, z)

def hand_near_ear(landmarks, hand_landmarks, w, h) -> bool:
    """
    Check if hand fingertips are near ear landmarks.
    CRITICAL: Add Y-coordinate filtering to prevent mouth area confusion:
    - Phone call: Hand in UPPER half of face (above nose level)
    - Smoking: Hand in LOWER half (mouth area)
    
    Asymmetric thresholds for 45-degree side-mounted camera:
    - Left ear (near camera): 5% of face height
    - Right ear (far from camera): 50% of face height (large for side angle)
    """
    try:
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        face_h = max(1.0, float(max(ys) - min(ys)))
        face_top = min(ys)
        face_bottom = max(ys)
    except Exception:
        face_h = float(h)
        face_top = 0
        face_bottom = h
    
    # Get nose landmark (landmark 1) as dividing line between upper/lower face
    try:
        nose_y = landmarks[1].y * h
    except Exception:
        nose_y = (face_top + face_bottom) / 2.0
    
    ear_l = np.array([landmarks[LEFT_EAR_TIP].x * w, landmarks[LEFT_EAR_TIP].y * h])
    ear_r = np.array([landmarks[RIGHT_EAR_TIP].x * w, landmarks[RIGHT_EAR_TIP].y * h])
    tips = [4, 8, 12, 16, 20]  # Hand fingertip landmarks
    
    # Asymmetric radius for 45-degree camera
    r_pix_left = 0.10 * face_h   # Left ear (closer to camera) - relaxed for easier detection
    r_pix_right = 0.60 * face_h  # Right ear (farther from camera) - maximum for side angle perspective
    
    # Use nose as filter boundary but add margin for right ear (allow 10% below nose for right ear)
    nose_filter_left = nose_y  # Strict for left ear
    nose_filter_right = nose_y + (0.10 * face_h)  # Relaxed for right ear (allow 10% below nose)
    
    near_l = 0
    near_r = 0
    for idx in tips:
        try:
            lm = hand_landmarks.landmark[idx]
        except Exception:
            continue
        hx, hy = lm.x * w, lm.y * h
        
        # Left ear: strict upper face only (above nose)
        if hy <= nose_filter_left:
            if np.hypot(hx - ear_l[0], hy - ear_l[1]) <= r_pix_left:
                near_l += 1
        
        # Right ear: allow slightly below nose (side camera angle)
        if hy <= nose_filter_right:
            if np.hypot(hx - ear_r[0], hy - ear_r[1]) <= r_pix_right:
                near_r += 1
    
    # Both ears: need only 1+ fingertip (easier detection for both sides)
    return (near_l >= 1) or (near_r >= 1)

def hand_near_face(face_center, hand_landmarks, shape, px: int = 200) -> bool:
    fcx, fcy = face_center
    ih, iw = shape[:2]
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * iw), int(lm.y * ih)
        if np.hypot(fcx - x, fcy - y) < px:
            return True
    return False

# YOLO utilities (Ultralytics or ONNXRuntime fallback)
import os as _os
import re as _re
import time as _time
import threading as _threading
import queue as _queue
from typing import Dict as _Dict, List as _List, Tuple as _Tuple

# Global unified print function - defined early for use in classes
_dms_print_func = None
def set_dms_print(func):
    global _dms_print_func
    _dms_print_func = func

def dms_print_global(category, message, frame_id=None):
    """Global print function callable from classes before main dms_print is defined"""
    if _dms_print_func:
        _dms_print_func(category, message, frame_id)
    else:
        # Fallback if not initialized
        if frame_id is not None:
            print(f"[{category}] Frame {frame_id}: {message}")
        else:
            print(f"[{category}] {message}")

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False
try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    ort = None
    _HAS_ORT = False

DEFAULT_NAMES = [
    "seatbelt_worn",
    "no_seatbelt",
    "cigarette_Hand",
    "cigarette_Mouth",
    "eye_Closed",
    "eye_Open",
    "no_Cigarette",
]

def parse_thresholds(th_str: str) -> dict:
    out: dict = {}
    if not th_str:
        return out
    for kv in th_str.split(','):
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        k = k.strip(); v = v.strip()
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

def _norm_name(s: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def _convert_results_to_dets(res, names: list, class_thresholds: dict, base_conf: float):
    dets = []
    try:
        r0 = res[0]
    except Exception:
        r0 = None
    if r0 is None or getattr(r0, 'boxes', None) is None or len(r0.boxes) == 0:
        return dets
    for b in r0.boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        cls = int(b.cls[0].item()) if b.cls is not None else -1
        conf = float(b.conf[0].item()) if b.conf is not None else 0.0
        name = names[cls] if 0 <= cls < len(names) else str(cls)
        thr = max(base_conf, class_thresholds.get(name, class_thresholds.get(_norm_name(name), 0.0)))
        if conf >= thr:
            dets.append((xyxy, name, conf))
    return dets

class YOLOWorker(_threading.Thread):
    def __init__(self, model_path: str, imgsz: int, conf: float, iou: float,
                 class_thresholds: dict, names: list, nms_topk: int = 300):
        super().__init__(daemon=True)
        self.queue: "_queue.Queue" = _queue.Queue(maxsize=1)
        self._stop_ev = _threading.Event()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.class_thresholds = class_thresholds
        self.names = names
        self.nms_topk = int(nms_topk) if nms_topk and int(nms_topk) > 0 else 300
        self.latest = None
        self.yolo = None
        self.ort_session = None
        self.ort_input_name = None
        self.ort_input_shape = None
        # Only use Ultralytics for PyTorch model files; ONNX will be handled by ORT directly
        if _HAS_YOLO and _os.path.isfile(model_path) and model_path.lower().endswith(('.pt', '.pth')):
            try:
                self.yolo = YOLO(model_path)
                if hasattr(self.yolo.model, 'names') and len(self.yolo.model.names) >= 7:
                    self.names = [self.yolo.model.names[i] for i in range(7)]
            except Exception as e:
                dms_print_global("YOLOWorker", f"Failed to load YOLO: {e}")
        if (self.yolo is None) and _HAS_ORT and _os.path.isfile(model_path) and model_path.lower().endswith('.onnx'):
            try:
                sess_opts = ort.SessionOptions()
                # CPU perf tweaks
                try:
                    sess_opts.intra_op_num_threads = max(1, _os.cpu_count() or 1)
                    sess_opts.inter_op_num_threads = 1
                    sess_opts.execution_mode = getattr(ort, 'ExecutionMode', None).SEQUENTIAL if hasattr(ort, 'ExecutionMode') else 0
                    if hasattr(ort, 'GraphOptimizationLevel'):
                        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                except Exception:
                    pass
                self.ort_session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
                self.ort_input_name = self.ort_session.get_inputs()[0].name
                self.ort_input_shape = self.ort_session.get_inputs()[0].shape
            except Exception as e:
                dms_print_global("YOLOWorker", f"Failed to load ONNXRuntime session: {e}")
                import traceback
                traceback.print_exc()

    def validate_model(self) -> bool:
        """Run a tiny dry-run to ensure the model is actually usable.
        Returns True if inference path executes without raising, False otherwise."""
        try:
            # Create a small dummy frame to keep validation fast
            sz = int(max(32, min(128, self.imgsz)))
            dummy = np.zeros((sz, sz, 3), dtype=np.uint8)
            if self.yolo is not None:
                _ = self.yolo.predict(source=dummy, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                      device='cpu', verbose=False)
                return True
            if self.ort_session is not None:
                _ = self._infer_onnx(dummy)
                return True
            return False
        except Exception as e:
            dms_print_global("YOLOWorker", f"Model validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def submit(self, frame):
        if self._stop_ev.is_set():
            return
        try:
            while True:
                self.queue.get_nowait()
        except Exception:
            pass
        try:
            self.queue.put_nowait(frame.copy())
        except Exception:
            pass

    def get_latest(self):
        return self.latest

    def stop(self):
        self._stop_ev.set()

    def run(self):
        if (self.yolo is None) and (self.ort_session is None):
            print("[YOLOWorker] YOLO not available; worker idle")
            return
        while not self._stop_ev.is_set():
            try:
                frame = self.queue.get(timeout=0.25)
            except Exception:
                continue
            try:
                if self.yolo is not None:
                    # Note: some Ultralytics versions don't support 'persist' kwarg in predict();
                    # keeping the model instance alive already provides persistence across calls.
                    res = self.yolo.predict(source=frame, imgsz=int(self.imgsz), conf=self.conf, iou=self.iou,
                                            device='cpu', verbose=False)
                    dets = _convert_results_to_dets(res, self.names, self.class_thresholds, self.conf)
                else:
                    dets = self._infer_onnx(frame)
                self.latest = (_time.time(), dets)
            except Exception as e:
                print("[YOLOWorker] inference error:", e)

    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        h, w = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw //= 2
        dh //= 2
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img_padded = cv2.copyMakeBorder(img_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
        return img_padded, r, dw, dh

    def _infer_onnx(self, frame):
        import numpy as _np
        if self.ort_session is None:
            return []
        img0 = frame
        # Use model's actual expected input size from ONNX metadata
        if self.ort_input_shape and len(self.ort_input_shape) == 4:
            img_size = int(self.ort_input_shape[2])  # Assumes NCHW format: [batch, channels, height, width]
        else:
            img_size = int(self.imgsz)
        # Preprocess input
        img, r, dw, dh = self._letterbox(img0, img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img_rgb.transpose(2, 0, 1)[None].astype('float32') / 255.0
        outputs = self.ort_session.run(None, {self.ort_input_name: x})
        dets = []
        nc = len(self.names)
        def as_2d(arr):
            return arr[0] if (arr.ndim == 3 and arr.shape[0] == 1) else arr
        if len(outputs) >= 2:
            A, B = outputs[0], outputs[1]
            A = as_2d(A); B = as_2d(B)
            boxes = scores = None
            for out in (A, B):
                if out.ndim == 3:
                    out = out[0]
                if out.ndim == 2:
                    if out.shape[1] == 4:
                        boxes = out
                    elif out.shape[0] == 4:
                        boxes = out.T
                    elif out.shape[1] == nc:
                        scores = out
                    elif out.shape[0] == nc:
                        scores = out.T
            if boxes is None or scores is None:
                return []
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - w / 2; y1 = cy - h / 2; x2 = cx + w / 2; y2 = cy + h / 2
            class_ids = _np.argmax(scores, axis=1)
            confs = scores[_np.arange(scores.shape[0]), class_ids]
            for i in range(boxes.shape[0]):
                conf = float(confs[i]); cid = int(class_ids[i])
                name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                thr = max(self.conf, self.class_thresholds.get(name, self.class_thresholds.get(_norm_name(name), 0.0)))
                if conf < thr:
                    continue
                dets.append([x1[i], y1[i], x2[i], y2[i], conf, cid])
        else:
            z = outputs[0]
            if z.ndim == 3 and z.shape[0] == 1:
                if z.shape[1] in (nc + 4, 84, 85):
                    z = _np.transpose(z, (0, 2, 1))
                z = z[0]
            if z.ndim == 2 and (z.shape[1] == (nc + 4)):
                boxes = z[:, :4]; scores = z[:, 4:]
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = cx - w / 2; y1 = cy - h / 2; x2 = cx + w / 2; y2 = cy + h / 2
                class_ids = _np.argmax(scores, axis=1)
                confs = scores[_np.arange(scores.shape[0]), class_ids]
                for i in range(z.shape[0]):
                    conf = float(confs[i]); cid = int(class_ids[i])
                    name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                    thr = max(self.conf, self.class_thresholds.get(name, self.class_thresholds.get(_norm_name(name), 0.0)))
                    if conf < thr:
                        continue
                    dets.append([x1[i], y1[i], x2[i], y2[i], conf, cid])
            elif z.ndim == 2 and z.shape[1] == 6:
                for i in range(z.shape[0]):
                    x1, y1, x2, y2, conf, cid = map(float, z[i])
                    cid = int(cid)
                    name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                    thr = max(self.conf, self.class_thresholds.get(name, self.class_thresholds.get(_norm_name(name), 0.0)))
                    if conf < thr:
                        continue
                    dets.append([x1, y1, x2, y2, conf, cid])
            else:
                if z.ndim == 2 and z.shape[0] in (nc + 4, 84, 85):
                    z = z.T
                if z.ndim == 2 and z.shape[1] == (nc + 4):
                    boxes = z[:, :4]; scores = z[:, 4:]
                    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = cx - w / 2; y1 = cy - h / 2; x2 = cx + w / 2; y2 = cy + h / 2
                    class_ids = _np.argmax(scores, axis=1)
                    confs = scores[_np.arange(scores.shape[0]), class_ids]
                    for i in range(z.shape[0]):
                        conf = float(confs[i]); cid = int(class_ids[i])
                        name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                        thr = max(self.conf, self.class_thresholds.get(name, self.class_thresholds.get(_norm_name(name), 0.0)))
                        if conf < thr:
                            continue
                        dets.append([x1[i], y1[i], x2[i], y2[i], conf, cid])
        if not dets:
            return []
        dets = _np.array(dets, dtype=_np.float32)
        keep = self._nms(dets[:, :4], dets[:, 4], self.iou, top_k=self.nms_topk)
        dets = dets[keep]
        out = []
        for x1_, y1_, x2_, y2_, conf_, cid_ in dets:
            x1o = max(0, int(round((x1_ - dw) / r)))
            y1o = max(0, int(round((y1_ - dh) / r)))
            x2o = max(0, int(round((x2_ - dw) / r)))
            y2o = max(0, int(round((y2_ - dh) / r)))
            name = self.names[int(cid_)] if 0 <= int(cid_) < len(self.names) else str(int(cid_))
            out.append(((x1o, y1o, x2o, y2o), name, float(conf_)))
        return out

    def _nms(self, boxes, scores, iou_thres=0.45, top_k=300):
        import numpy as _np
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0 and len(keep) < top_k:
            i = order[0]
            keep.append(i)
            xx1 = _np.maximum(x1[i], x1[order[1:]])
            yy1 = _np.maximum(y1[i], y1[order[1:]])
            xx2 = _np.minimum(x2[i], x2[order[1:]])
            yy2 = _np.minimum(y2[i], y2[order[1:]])
            w = _np.maximum(0.0, xx2 - xx1 + 1)
            h = _np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = _np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        return keep


# -------------------------------------------------------------
# MediaPipeWorker: runs face_mesh, hands, pose in a separate thread
# -------------------------------------------------------------
class MediaPipeWorker(_threading.Thread):
    """Background worker to offload (hands, pose) from main thread.

    Face mesh was removed from this worker to avoid frame-to-landmark lag
    (overlay misalignment). Hands and pose remain async for performance.
    """
    def __init__(self, face_mesh, hands, pose, args):
        super().__init__(daemon=True)
        import queue as _q
        self.queue: "_q.Queue" = _q.Queue(maxsize=1)
        self._stop_ev = _threading.Event()
        # Keep reference (unused now) so external calls not broken; could be None.
        self.face_mesh = None  # face mesh runs inline in main loop
        self.hands = hands
        self.pose = pose
        self.args = args
        self.latest = None  # tuple: (None, hand_result, pose_result)
        # face mesh caching removed (handled inline)
        self.last_hands_result = None
        self.last_pose_result = None
        self.fid = 0

    def submit(self, frame):
        if self._stop_ev.is_set():
            return
        # Drop any stale frame to keep only the newest
        try:
            while True:
                self.queue.get_nowait()
        except Exception:
            pass
        try:
            self.queue.put_nowait(frame.copy())
        except Exception:
            pass

    def get_latest(self):
        return self.latest

    def stop(self):
        self._stop_ev.set()

    def run(self):
        import cv2 as _cv2
        while not self._stop_ev.is_set():
            try:
                frame = self.queue.get(timeout=0.25)
            except Exception:
                continue
            try:
                h, w = frame.shape[:2]
                mp_scale = float(getattr(self.args, 'mp_scale', 1.0)) if getattr(self.args, 'mp_scale', 1.0) else 1.0
                if mp_scale != 1.0:
                    small = _cv2.resize(frame, (max(1, int(w * mp_scale)), max(1, int(h * mp_scale))), interpolation=_cv2.INTER_AREA)
                else:
                    small = frame
                rgb_small = _cv2.cvtColor(small, _cv2.COLOR_BGR2RGB)

                face_res = None  # always None here (processed inline in main)

                # Hands (stride)
                if self.hands is not None:
                    if (self.fid % max(1, int(self.args.hands_stride))) == 0:
                        hand_res = self.hands.process(rgb_small)
                        self.last_hands_result = hand_res
                    else:
                        hand_res = self.last_hands_result

                # Pose (stride)
                if self.pose is not None:
                    try:
                        if (self.fid % max(1, int(self.args.pose_stride))) == 0:
                            pose_res = self.pose.process(rgb_small)
                            self.last_pose_result = pose_res
                        else:
                            pose_res = self.last_pose_result
                    except Exception:
                        pose_res = self.last_pose_result

                self.latest = (None, hand_res, pose_res)
                self.fid += 1
            except Exception:
                # Keep worker alive on errors
                pass

def main():
    # ============================================================
    # OPTM MEDIA SOLUTION - DRIVER DROWSINESS MONITORING SYSTEM
    # ============================================================
    from datetime import datetime

    # Print company branding banner
    print("\n" + "="*70)
    print("")
    print("   ╔══════════════════════════════════════════════════════════╗")
    print("   ║                                                          ║")
    print("   ║   ██████╗ ██████╗ ████████╗███╗   ███╗                   ║")
    print("   ║  ██╔═══██╗██╔══██╗╚══██╔══╝████╗ ████║                   ║")
    print("   ║  ██║   ██║██████╔╝   ██║   ██╔████╔██║                   ║")
    print("   ║  ██║   ██║██╔═══╝    ██║   ██║╚██╔╝██║                   ║")
    print("   ║  ╚██████╔╝██║        ██║   ██║ ╚═╝ ██║                   ║")
    print("   ║   ╚═════╝ ╚═╝        ╚═╝   ╚═╝     ╚═╝                   ║")
    print("   ║                                                          ║")
    print("   ║            OptM Media Solution Pvt Ltd                   ║")
    print("   ║                                                          ║")
    print("   ║      Driver Drowsiness Monitoring System (DMS)           ║")
    print("   ║                                                          ║")
    print("   ╚══════════════════════════════════════════════════════════╝")
    print("")
    print("="*70 + "\n")
    
    ap = argparse.ArgumentParser()
    # MediaPipe thresholds/options
    ap.add_argument('--ear_threshold', type=float, default=0.180)
    ap.add_argument('--eye_closed_frames_threshold', type=int, default=9)
    # Risk scoring and sessions (optional)
    ap.add_argument('--risk_on', action='store_true', help='Enable composite risk scoring and per-trip session logging')
    ap.add_argument('--risk_weights', type=str, default='drowsiness=3,phone=2,gaze=2,seatbelt=3,smoking=1,texting=2,medical=3', help='Comma list weights for events (e.g., drowsiness=3,phone=2)')
    ap.add_argument('--risk_update_interval', type=float, default=1.0, help='Seconds between risk accumulation updates')
    ap.add_argument('--session_auto_end_gap', type=float, default=900.0, help='End active session if no recognized face for this many seconds (default 15m)')
    # Friendly alias for handover use-case (maps to session_auto_end_gap)
    ap.add_argument('--handover_gap', type=float, help='Alias for --session_auto_end_gap (seconds). Recommended 5-10s to tolerate quick driver swaps.')
    ap.add_argument('--blink_rate_threshold', type=int, default=5)
    ap.add_argument('--mar_threshold', type=float, default=0.6)
    ap.add_argument('--yawn_threshold', type=int, default=3)
    ap.add_argument('--frame_width', type=int, default=1280)
    ap.add_argument('--frame_height', type=int, default=720)
    ap.add_argument('--gaze_deviation_threshold', type=float, default=0.05)
    ap.add_argument('--scale_factor', type=float, default=1.0)
    ap.add_argument('--head_turn_threshold', type=float, default=0.15, help='Threshold for head turn detection (0.15 = tolerant for side-mounted cameras, 0.08 = strict for front cameras)')
    ap.add_argument('--hand_near_face_px', type=int, default=200)
    ap.add_argument('--calibration_time', type=int, default=10, help='Auto-calibration time (10s recommended for any camera angle)')
    ap.add_argument('--no_face_display', action='store_true')
    ap.add_argument('--no_mesh_display', action='store_true')
    # Optional retina-style overlay (off by default to avoid changing UI)
    ap.add_argument('--retina_overlay', action='store_true', help='draw thin white gaze lines and show State/EAR/Blinks HUD')
    # Performance controls
    ap.add_argument('--mp_scale', type=float, default=1.0, help='downscale factor for MediaPipe processing (e.g., 0.75 or 0.5)')
    ap.add_argument('--face_stride', type=int, default=1, help='run face mesh every N frames (1 = every frame)')
    ap.add_argument('--hands_stride', type=int, default=1, help='run hands every N frames')
    ap.add_argument('--pose_stride', type=int, default=1, help='run pose every N frames')
    ap.add_argument('--cv_threads', type=int, default=0, help='set OpenCV number of threads (0=leave default)')
    # Medical distress detection (optional – off by default)
    ap.add_argument('--medical_on', action='store_true', help='Enable medical distress heuristics (phase 1: chest clutching)')
    ap.add_argument('--chest_clutch_min_secs', type=float, default=4.0, help='Seconds a hand must remain on chest to trigger medical alert')
    ap.add_argument('--chest_roi_expand', type=float, default=0.10, help='Expand chest ROI by this fraction for chest-clutching detection')
    ap.add_argument('--chest_stationary_eps', type=float, default=0.03, help='Max normalized hand movement (per second) allowed to consider it stationary')
    ap.add_argument('--medical_emit_interval', type=float, default=2.0, help='Re-emit medical alert every N seconds while condition holds')
    ap.add_argument('--medical_cooldown_secs', type=float, default=6.0, help='Cooldown after a chest-clutch episode ends before detecting again')

    # YOLO options
    ap.add_argument('--model', type=str, default=os.path.join(os.path.dirname(__file__), 'best_dyn.onnx'))
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--yolo_conf', type=float, default=0.25)
    ap.add_argument('--yolo_iou', type=float, default=0.45)
    ap.add_argument('--yolo_skip', type=int, default=1, help='Run YOLO every N frames (1 = every frame)')
    ap.add_argument('--class_thresholds', type=str, default='', help='per-class conf: name=0.35,name2=0.4')
    ap.add_argument('--show_boxes', action='store_true', help='draw YOLO boxes and labels (alias of --draw_boxes)')
    ap.add_argument('--draw_boxes', action='store_true', help='draw YOLO boxes and labels')
    ap.add_argument('--yolo_labels_overlay', action='store_true', help='overlay raw YOLO class names as on-screen labels')
    ap.add_argument('--yolo_status_overlay', action='store_true', help='show a small bottom-left YOLO status line (off by default)')
    ap.add_argument('--yolo_overlay_conf', action='store_true', help='show confidence in yolo_labels_overlay (hidden by default)')
    ap.add_argument('--yolo_overlay_include_belt_smoke', action='store_true', help='include seatbelt and cigarette classes in yolo_labels_overlay (excluded by default)')
    ap.add_argument('--box_label_include_belt_smoke', action='store_true', help='draw text on seatbelt/cigarette boxes (default: only draw rectangles, no text)')
    ap.add_argument('--show_cig_boxes', action='store_true', help='draw rectangles for cigarette_* (default: hidden, only HUD text shown)')
    ap.add_argument('--debug_yolo', action='store_true')
    ap.add_argument('--no_yolo_gating', action='store_true', help='disable ROI gating for YOLO events')
    ap.add_argument('--nms_topk', type=int, default=300, help='max boxes to keep after NMS (smaller can be faster)')
    # Allow showing MediaPipe phone/texting alerts even when smoking is detected
    ap.add_argument('--no_smoking_suppression', action='store_true', help='do not suppress phone/texting alerts during smoking frames')
    # YOLO smoking/face ROI tuning
    ap.add_argument('--cig_face_roi_expand', type=float, default=0.8, help='expand factor around face bbox for smoking gating (was 0.6)')
    ap.add_argument('--cig_mouth_radius', type=float, default=0.55, help='mouth-centered acceptance radius as fraction of face height')
    ap.add_argument('--cig_hand_radius', type=float, default=0.65, help='hand-centered acceptance radius as fraction of face height')
    # Extra filters to suppress finger-only false positives (tunable, conservative defaults)
    ap.add_argument('--cig_min_aspect_hand', type=float, default=1.8, help='min aspect ratio (long/short) for hand cigarette bbox')
    ap.add_argument('--cig_min_aspect_mouth', type=float, default=1.6, help='min aspect ratio (long/short) for mouth cigarette bbox')
    ap.add_argument('--cig_max_area_frac_hand', type=float, default=0.35, help='max bbox area as fraction of face area for hand cigarette')
    ap.add_argument('--cig_max_area_frac_mouth', type=float, default=0.30, help='max bbox area as fraction of face area for mouth cigarette')
    ap.add_argument('--cig_min_brightness_hand', type=int, default=110, help='min mean grayscale brightness for hand cigarette ROI (0-255)')
    ap.add_argument('--cig_min_brightness_mouth', type=int, default=100, help='min mean grayscale brightness for mouth cigarette ROI (0-255)')
    # Optional whiteness gating (helps reject pens/fingers): low saturation + bright pixels
    ap.add_argument('--cig_max_saturation_hand', type=int, default=80, help='max HSV saturation for white-ish pixels (hand) [0-255]')
    ap.add_argument('--cig_max_saturation_mouth', type=int, default=80, help='max HSV saturation for white-ish pixels (mouth) [0-255]')
    ap.add_argument('--cig_min_white_frac_hand', type=float, default=0.10, help='min fraction of white-ish pixels in ROI (hand) [0..1]')
    ap.add_argument('--cig_min_white_frac_mouth', type=float, default=0.08, help='min fraction of white-ish pixels in ROI (mouth) [0..1]')
    ap.add_argument('--cig_enable_mouth_whiteness', action='store_true', help='apply whiteness gating at mouth (off by default to avoid missing real cigarettes)')
    # Near-mouth relaxation for hand whiteness (helps detect small tip near lips)
    ap.add_argument('--cig_relax_dist_norm', type=float, default=0.28, help='within this normalized distance to mouth, relax hand whiteness requirement')
    ap.add_argument('--cig_relax_white_frac_hand', type=float, default=0.06, help='min white frac for hand when very close to mouth (normalized by cig_relax_dist_norm)')
    ap.add_argument('--cig_relax_brightness_hand', type=int, default=95, help='min brightness for hand ROI when very close to mouth')
    # Near-mouth easy accept for mouth detections
    ap.add_argument('--cig_mouth_easy_norm', type=float, default=0.18, help='within this normalized distance to mouth, accept mouth detections without extra filters')
    # Smoking escalation + hysteresis
    ap.add_argument('--cig_hand_to_mouth_dist_norm', type=float, default=0.22, help='if hand-cig bbox center is within this fraction of face height to mouth center, escalate to Mouth')
    ap.add_argument('--cig_hold_frames_mouth', type=int, default=8, help='hold mouth state for N frames after signal disappears')
    ap.add_argument('--cig_hold_frames_hand', type=int, default=5, help='hold hand state for N frames after signal disappears')
    # Smoking status stability tuning
    ap.add_argument('--cig_none_min_frames', type=int, default=2, help='frames to confirm No Cigarette to avoid quick false clears')
    ap.add_argument('--cig_decay_frames', type=int, default=15, help='frames with no smoking/no-cigar detections before decaying to No Cigarette (face visible)')

    # Seatbelt voting
    ap.add_argument('--seatbelt_conf_margin', type=float, default=0.08)
    ap.add_argument('--seatbelt_vote_len', type=int, default=20)
    ap.add_argument('--seatbelt_vote_threshold', type=int, default=4)
    ap.add_argument('--seatbelt_emit_interval', type=float, default=0.1, help='seconds between periodic seatbelt status re-emission')
    # Seatbelt transition/hold tuning
    ap.add_argument('--seatbelt_off_min_frames', type=int, default=1, help='frames needed to confirm Worn→NO quick transition')
    ap.add_argument('--seatbelt_on_min_frames', type=int, default=1, help='frames needed to confirm NO→Worn quick transition')
    ap.add_argument('--seatbelt_hold_secs', type=float, default=0.20, help='hold last seatbelt status when detections momentarily drop (occlusion)')
    # Smoking status periodic emit
    ap.add_argument('--smoke_emit_interval', type=float, default=0.8, help='seconds between periodic smoking status re-emission')

    # Head-turn direction control
    ap.add_argument('--flip_yaw_sign', action='store_true', help='flip yaw sign for Left/Right labeling')

    # Diagnostics
    ap.add_argument('--fps_interval', type=int, default=60, help='print FPS every N frames (0=disable)')
    ap.add_argument('--threaded_yolo', action='store_true', help='compat: YOLO runs in a background thread by default')
    ap.add_argument('--quiet', action='store_true', help='suppress console prints/logs')
    ap.add_argument('--fast_preset_imx8', action='store_true', help='optimize defaults for i.MX8 CPU-only (no NPU)')
    ap.add_argument('--keep_mesh_display', action='store_true', help='override preset to keep MediaPipe mesh drawing visible')
    # Remote TCP frame sending
    ap.add_argument('--remote_server', type=str, default=None, help='Remote server IP address (e.g., 192.168.1.100). If set, sends frames to remote server via TCP')
    ap.add_argument('--remote_port', type=int, default=5000, help='Remote server port (default: 5000)')
    ap.add_argument('--no_stream', action='store_true', help='Disable video frame streaming (only log data/alerts to terminal). Use with --remote_server for logs-only mode.')
    # Camera selection
    ap.add_argument('--camera_device', type=str, default='auto', help="Camera device (e.g., /dev/video0, 0, or 'auto' for auto-detect)")
    ap.add_argument('--use_gstreamer', type=bool, default=True, help="Use GStreamer for hardware accelerated camera capture (default: True for production)")
    ap.add_argument('--crop_center', type=float, default=0.0, help="Center crop percentage (0.0-0.5). E.g., 0.3 crops 30%% from each edge")
    # Welcome-back / heartbeat configuration
    ap.add_argument('--welcome_back_gap_secs', type=float, default=15.0, help='gap in seconds to consider a "Welcome back" for the same driver')
    # Driver authentication
    ap.add_argument('--enable_auth', action='store_true', help='Enable driver face authentication at startup')
    ap.add_argument('--auth_timeout', type=int, default=10, help='Authentication timeout in seconds (default: 10s)')
    ap.add_argument('--auth_detector', type=str, default='scrfd_500m_full_int8_vela.tflite', help='SCRFD detector model for authentication')
    ap.add_argument('--auth_recognizer', type=str, default='fr_int8_velaS.tflite', help='Face recognition model for authentication')
    ap.add_argument('--auth_database', type=str, default='drivers.json', help='Driver database path for authentication')
    # Anti-spoofing liveness detection
    ap.add_argument('--enable_liveness', action='store_true', default=True, help='Enable liveness detection to prevent photo/video spoofing (default: True)')
    ap.add_argument('--liveness_blink_min', type=int, default=2, help='Minimum blinks required for liveness (default: 2)')
    ap.add_argument('--liveness_motion_thresh', type=float, default=0.8, help='Micro-motion threshold for liveness (default: 0.8)')
    # Optimization/selection controls
    ap.add_argument('--prefer_onnx', action='store_true', help='prefer an ONNX model over .pt when both are available')
    ap.add_argument('--auto_yolo_skip', action='store_true', help='auto-adjust YOLO frame skip based on FPS (1..3)')
    ap.add_argument('--auto_imgsz', action='store_true', help='auto-tune imgsz based on frame size for CPU')
    # Robustness controls for cigarette detections
    ap.add_argument('--require_face_for_cig', action='store_true', default=True, help='only accept cigarette_* detections when a face is visible')
    ap.add_argument('--cig_max_area_frame_frac', type=float, default=0.18, help='suppress cigarette_* boxes that cover too much of the frame (near-camera false positives)')

    args = ap.parse_args()
    # Normalize alias
    try:
        if getattr(args, 'handover_gap', None) is not None:
            args.session_auto_end_gap = float(args.handover_gap)
    except Exception:
        pass
    # Force face alignment always on irrespective of CLI flag (user request)
    try:
        args.face_align = True
    except Exception:
        pass
    if '--face_align' not in sys.argv and '--no_face_align' not in sys.argv:
        try:
            print('[FaceRec] Face alignment forced ON by default.')
        except Exception:
            pass

    # Early action: list drivers and exit (JSON-only; SQLite removed)
    if getattr(args, 'list_drivers', False):
        try:
            from face_id import load_profiles_from_file
            path = getattr(args, 'drivers_file', None) or 'drivers.json'
            if os.path.isfile(path):
                profiles = load_profiles_from_file(path)
                if profiles:
                    print(f"[Drivers] {len(profiles)} profiles (file: {os.path.basename(path)})")
                    for p in profiles:
                        print(f"  ID={p.driver_id}  Name={p.name}  Dim={p.embedding.shape[0]}")
                else:
                    print(f"[Drivers] No profiles found in file '{path}'")
            else:
                print(f"[Drivers] No drivers file found at '{path}'")
        except Exception as e:
            print(f"[Drivers] Listing failed: {e}")
        return

    # Optional fast preset for i.MX8 or similar CPU-only targets (no NPU)
    if getattr(args, 'fast_preset_imx8', False):
        try:
            args.imgsz = min(int(args.imgsz), 320)
        except Exception:
            args.imgsz = 320
        try:
            args.yolo_skip = max(int(args.yolo_skip), 2)
        except Exception:
            args.yolo_skip = 2
        # Reduce console and overlay costs
        args.fps_interval = 0
        args.no_mesh_display = True
        args.debug_yolo = False
        # MediaPipe downscale + stride to save CPU
        try:
            if not hasattr(args, 'mp_scale') or args.mp_scale is None:
                args.mp_scale = 0.75
            else:
                args.mp_scale = min(1.0, max(0.5, float(args.mp_scale)))
        except Exception:
            args.mp_scale = 0.75
        try:
            args.face_stride = max(1, int(getattr(args, 'face_stride', 2)))
            args.hands_stride = max(1, int(getattr(args, 'hands_stride', 2)))
            args.pose_stride = max(1, int(getattr(args, 'pose_stride', 3)))
        except Exception:
            args.face_stride, args.hands_stride, args.pose_stride = 2, 2, 3
        # Slightly lower NMS top-k to reduce postproc cost
        try:
            args.nms_topk = max(100, int(getattr(args, 'nms_topk', 200)))
        except Exception:
            args.nms_topk = 200
        # Allow user to keep meshes visible despite preset
        if getattr(args, 'keep_mesh_display', False):
            args.no_mesh_display = False

    try:
        cv2.setUseOptimized(True)
        if getattr(args, 'cv_threads', 0):
            cv2.setNumThreads(int(args.cv_threads))
    except Exception:
        pass

    # Unified print function for all DMS messages - as requested by colleague (defined early)
    def dms_print(category, message, frame_id=None):
        """
        Ultra-fast unified print function for all DMS output
        Uses direct sys.stdout write for minimum latency
        """
        import sys
        if getattr(args, 'quiet', False):
            return
        if frame_id is not None:
            msg = f"[{category}] Frame {frame_id}: {message}\n"
        else:
            msg = f"[{category}] {message}\n"
        sys.stdout.write(msg)
        sys.stdout.flush()
    
    # Register global print function for classes
    set_dms_print(dms_print)

    # Initialize remote TCP frame sender
    frame_sender = None
    no_stream_mode = getattr(args, 'no_stream', False)
    
    if getattr(args, 'remote_server', None):
        # Remote streaming mode (TCP to external server)
        if no_stream_mode:
            # Don't initialize frame_sender - keep it None
            pass
        else:
            frame_sender = FrameSender(args.remote_server, getattr(args, 'remote_port', 5000) or 5000)
            if not frame_sender.connect():
                pass
    else:
        if no_stream_mode:
            pass

    # Driver Authentication (if enabled)
    authenticated_driver = None
    driver_id = None
    
    if getattr(args, 'enable_auth', False) and _HAS_AUTH:
        optm_print("Authentication", "Driver authentication enabled")
        try:
            auth_success, auth_name, auth_id = quick_authenticate(
                detector_path=args.auth_detector,
                recognizer_path=args.auth_recognizer,
                database_path=args.auth_database,
                camera_id=0 if args.camera_device == 'auto' else int(args.camera_device) if args.camera_device.isdigit() else 0,
                timeout_seconds=args.auth_timeout,
                enable_liveness=getattr(args, 'enable_liveness', True),
                liveness_blink_min=getattr(args, 'liveness_blink_min', 2),
                liveness_motion_thresh=getattr(args, 'liveness_motion_thresh', 0.8)
            )
            
            if auth_success:
                authenticated_driver = auth_name
                driver_id = auth_id
                optm_print("Authentication", f"✓ Driver authenticated: {auth_name} (ID: {auth_id})")
            else:
                optm_print("Authentication", "⚠ Proceeding in unauthorized mode")
            
            # Force cleanup and free memory
            import gc
            gc.collect()
            optm_print("Authentication", "Models unloaded - starting DMS monitoring")
            
        except Exception as e:
            optm_print("Authentication", f"✗ Authentication failed: {e}")
            optm_print("Authentication", "Proceeding in unauthorized mode")
    else:
        optm_print("Authentication", "Driver authentication disabled")
    
    # Production Configuration Summary
    if authenticated_driver:
        optm_print("Driver", f"{authenticated_driver} (ID: {driver_id})")
    
    # MediaPipe solutions - simple inline processing like DMSv8
    face_mesh, hands, pose, mp_drawing, hand_connections, face_mesh_connections = create_solutions()
    
    optm_print("Calibration", f"Starting {args.calibration_time}s calibration period...")
    optm_print("Calibration", "Look straight ahead and keep face visible")

    # Risk/session state
    risk_enabled = bool(getattr(args, 'risk_on', False))
    risk_weights = {}
    try:
        for kv in str(getattr(args, 'risk_weights', '') or '').split(','):
            if not kv.strip():
                continue
            k, v = kv.split('='); risk_weights[k.strip()] = int(v)
    except Exception:
        # default weights
        risk_weights = {'drowsiness':3, 'phone':2, 'gaze':2, 'seatbelt':3, 'smoking':1, 'texting':2, 'medical':3}
    last_risk_update_ts = time.time()
    current_risk_level = 0

    # YOLO worker
    class_thresholds = parse_thresholds(args.class_thresholds)
    class_thresholds = {**class_thresholds, **{_norm_name(k): v for k, v in class_thresholds.items()}}
    names = DEFAULT_NAMES

    base_dir = os.path.dirname(__file__)
    candidate_paths = []
    # Always consider explicit user model first
    if args.model:
        candidate_paths.append(args.model)
    # Then fallback candidates in a reasonable order
    candidate_paths += [
        os.path.join(base_dir, 'best_dyn.onnx'),
        os.path.join(base_dir, 'best.onnx'),
        os.path.join(base_dir, 'best.pt'),
    ]
    # Deduplicate while preserving order
    seen = set()
    candidate_paths = [p for p in candidate_paths if (p not in seen and (seen.add(p) or True))]

    yolo_worker = None
    selected_model = None
    for p in candidate_paths:
        if not os.path.isfile(p):
            continue
        # Only try compatible loader per extension inside YOLOWorker
        worker = YOLOWorker(p, args.imgsz, args.yolo_conf, args.yolo_iou, class_thresholds, names, nms_topk=args.nms_topk)
        if worker.validate_model():
            yolo_worker = worker
            selected_model = p
            yolo_worker.start()
            break
        else:
            # Explicitly stop thread if it started anything (it shouldn't block)
            try:
                worker.stop()
            except Exception:
                pass
            if not args.quiet:
                dms_print("YOLO", f"Skipping unusable model: {p}")
    if yolo_worker is None and not args.quiet:
        dms_print("YOLO", "No usable YOLO model found (tried: " + ", ".join(candidate_paths) + ") — running MediaPipe only")

    # Camera initialization - Try GStreamer first (hardware accelerated), fallback to OpenCV
    try:
        if args.camera_device == 'auto':
            device_num = find_available_camera()
            if device_num is None:
                dms_print("ERROR", "No camera available")
                dms_print("INFO", "Try: ls -l /dev/video* or specify manually with --camera_device")
                exit(1)
        else:
            if '/dev/video' in args.camera_device:
                device_num = int(args.camera_device.split('video')[-1])
            else:
                device_num = int(args.camera_device)

        use_gstreamer = getattr(args, 'use_gstreamer', True)  # Default TRUE for production
        cap = None

        # ---------- UPDATED GStreamer block ----------
        if use_gstreamer:
            try:
                gst_pipeline = (
                    f"v4l2src device=/dev/video{device_num} ! "
                    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
                    "videoconvert ! "
                    "video/x-raw,format=BGR ! "
                    "appsink drop=true max-buffers=1 sync=false"
                )

                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

                if cap is not None and cap.isOpened():
                    # Retry a few reads so we don't fallback due to a slow first frame
                    ok = False
                    frame = None
                    for _ in range(10):
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            break

                    if ok and frame is not None:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # If your code needs this first frame later, store it; otherwise ignore.
                    else:
                        cap.release()
                        cap = None
                else:
                    cap = None

            except Exception as e:
                cap = None
        # ---------- END UPDATED GStreamer block ----------

        # Fallback to OpenCV V4L2 if GStreamer failed or disabled
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(device_num, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Request YUYV + 640x480 (driver may honor it; if not, it will fallback internally)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Optional: verify first read in fallback too (helps debugging)
            ok, frame = cap.read()

        if not cap.isOpened():
            dms_print("ERROR", f"Cannot open camera {device_num}")
            dms_print("INFO", "Try: ls -l /dev/video* to find your camera")
            dms_print("INFO", "Available cameras: 0, 1, 2, 3.")
            exit(1)

    except Exception as e:
        dms_print("ERROR", f"Camera initialization failed: {e}")
        exit(1)
        
     


    # State
    active_alerts = defaultdict(lambda: time.time())
    overlay_queue = []  # Queue for delayed overlay rendering (console prints first)
    OVERLAY_DELAY = 0.5  # 500ms delay before showing on screen (console is INSTANT at 0ms)
    ALERT_DURATION = 3
    eye_closure_counter = 0
    blink_counter = 0
    blink_total = 0
    blink_timer = time.time()
    yawn_counter = 0
    mar_deque = deque(maxlen=30)
    droop_active_since = None
    calibration_mode = True
    calibration_start_time = time.time()
    calibration_duration = args.calibration_time
    gaze_center = 0.5
    head_center_x = 0.5
    head_center_y = 0.5
    
    # Collect multiple samples during calibration for robust baseline (works better for angled cameras)
    calib_gaze_x_samples = deque(maxlen=600)
    calib_head_x_samples = deque(maxlen=600)
    calib_head_y_samples = deque(maxlen=600)
    calib_min_samples = 15
    countdown = int(calibration_duration)
    prev_eye_state = None
    
    # Eye closure tracking for duration, speed, and level
    eye_close_start_time = None  # When eyes started closing
    eye_close_duration = 0.0      # How long eyes have been closed (seconds)
    eye_closure_latency_logged = False  # Track if we already logged latency for this closure episode
    kss7_latency_logged = False   # Track if we already logged latency for KSS >= 7.0 episode
    buzzer_triggered = False      # Track if buzzer already fired for this closure event
    prev_ear = None               # Previous frame EAR for speed calculation
    eye_closure_speed = 0.0       # Rate of EAR change (units/second)
    prev_frame_time = time.time() # For calculating actual FPS
    frame_time_delta = 0.033      # Estimated time per frame (default ~30fps)
    sm_head_x_signed = 0.0
    sm_head_y_signed = 0.0
    sm_head_x_offset = 0.0
    sm_head_y_offset = 0.0

    # Simple 2D gaze direction detection (direct iris position comparison)
    # GAZE_THRESHOLD now in gaze detection code (0.025 left / 0.012 right)
    GAZE_CONFIRM_FRAMES = 3  # Require 3 consecutive frames for accuracy
    gaze_left_confirm_streak = 0
    gaze_right_confirm_streak = 0
    gaze_alerted_left = False
    gaze_alerted_right = False
    GAZE_ALERT_COOLDOWN = 5.0  # 5 seconds between re-alerts (reduced from 10s for better responsiveness)
    last_gaze_left_alert_time = 0.0
    last_gaze_right_alert_time = 0.0

    DROOP_SUSTAIN_SECONDS = 2.0
    HEAD_TURN_SUSTAIN_SECONDS = 5.0  # Head turn must be sustained for 5 seconds to trigger distraction
    head_turn_active_since = None  # Track when head turn started

    yolo_counters = {'smoking': 0}
    last_alert_times = defaultdict(lambda: 0.0)
    YOLO_ALERT_COOLDOWN = 2.0
    SEATBELT_ALERT_COOLDOWN = 1.0
    last_smoking_debug = 0.0

    face_bbox = None

    seatbelt_history = deque(maxlen=args.seatbelt_vote_len)
    no_seatbelt_streak = 0
    seatbelt_worn_streak = 0

    seatbelt_status = 'unknown'
    SEATBELT_EMIT_INTERVAL = args.seatbelt_emit_interval
    last_seatbelt_emit = 0.0
    # Fast transition helper for Worn -> NO (conservative, 2-frame confirm)
    seatbelt_off_trans_streak = 0
    # Fast transition helper for NO -> Worn
    seatbelt_on_trans_streak = 0
    # Track when we last saw any seatbelt-related detection (for optional hold behavior)
    last_seen_seatbelt_ts = 0.0

    # Smoking status stabilization and periodic emit
    smoke_status = 'unknown'   # one of: 'mouth','hand','none','unknown'
    smoking_mouth_streak = 0
    nocig_streak = 0
    nosignal_streak = 0
    SMOKE_EMIT_INTERVAL = args.smoke_emit_interval
    last_smoke_emit = 0.0
    # Holds to reduce flip-flop
    mouth_hold_left = 0
    # Grace window to avoid 'No Cigarette' right after smoking evidence
    smoke_recent_until_ts = 0.0
    
    # NEW SMOKING DETECTION: Simple counter for hand-to-mouth movements
    hand_mouth_counter = 0  # Count how many times hand moved to mouth
    SMOKING_CYCLE_THRESHOLD = 2  # Need 2+ hand-to-mouth movements to confirm smoking (filters false positives)
    SMOKING_CYCLE_WINDOW = 7.0  # Within 7 seconds
    hand_near_mouth_state = False  # Current state: is hand near mouth?
    smoking_alerted = False  # Once-only alert flag
    
    # Gaze tracking (45-degree side camera, 10cm above - CUSTOM CENTER for your setup)
    gaze_tracker = None
    if _HAS_GAZE_TRACKER:
        gaze_tracker = GazeTracker(
            horizontal_threshold=0.12,  # Tolerance around center point
            vertical_threshold=0.20,    # Tolerance around vertical center
            horizontal_center=0.35,     # YOUR actual center (not 0.50!) for 45° side camera
            vertical_center=0.60,       # YOUR vertical center (camera above eye level)
            smoothing_frames=2,         # Faster response
            flip_horizontal=True,       # Flip for front-facing camera
            flip_vertical=False         # Don't flip vertical
        )
    
    # Medical distress (chest clutching) state
    medical_enabled = bool(getattr(args, 'medical_on', False))
    chest_clutch_started_ts = 0.0
    chest_clutch_active = False
    last_medical_emit = 0.0
    last_chest_clutch_end_ts = 0.0
    prev_hand_centers_px = []  # list[(x_px,y_px)] aligned to current order of detected hands

    # Phone/texting detection with duration-based logic
    # NEW: Require sustained hand-near-ear for 5 seconds
    mobile_call_start_time = None  # Track when hand first went near ear
    MOBILE_CALL_DURATION = 5.0 # 5 seconds sustained = confirmed call
    mobile_call_alerted = False
    PHONE_ALERT_COOLDOWN = 30.0  # 30 seconds cooldown between re-alerts
    last_mobile_call_alert_time = 0.0
    
    texting_confirm_streak = 0
    PHONE_CONFIRM_FRAMES = 5  # require 5 consecutive frames before alert (~150ms)
    texting_alerted = False
    last_texting_alert_time = 0.0
    
    # Single-handed texting detection
    texting_hand_confirm_streak = 0
    TEXTING_HAND_CONFIRM_FRAMES = 3  # ~90ms at 2 FPS
    texting_hand_alerted = False
    last_texting_hand_alert_time = 0.0
    TEXTING_HAND_ALERT_COOLDOWN = 8.0  # 8 seconds between alerts

    frame_count = 0
    last_t = time.time()
    # Dynamic throttle state
    dynamic_skip = max(1, int(args.yolo_skip))
    fps_ema = None

    fid = 0
    last_yolo_ts = 0.0
    last_yolo_dets = []
    # Status helper: track if YOLO is producing results
    last_yolo_result_time = 0.0
    # Burst mode: after smoking evidence, run YOLO every frame briefly for snappier updates
    burst_until_ts = 0.0
    # Adaptive seatbelt skip: run every frame until stable, then relax to save CPU
    seatbelt_stable_frames = 0  # consecutive frames with same seatbelt status
    seatbelt_prev_status = 'unknown'
    SEATBELT_STABLE_THRESHOLD = 30  # frames needed to consider status stable (1 second at 30fps)
    SEATBELT_RELAXED_SKIP = 15  # skip frames when seatbelt status is stable
    
    # YOLO Cycling: Run every 10 seconds
    YOLO_INTERVAL = 10.0  # Run YOLO every 10 seconds
    last_yolo_submit_time = 0.0  # Track last submission time
    
    # Seatbelt alert limiting: alert 2 times, then sleep
    seatbelt_worn_print_count = 0
    no_seatbelt_print_count = 0
    SEATBELT_MAX_ALERTS = 2  # Alert 2 times then sleep
    no_seatbelt_alert_count = 0  # Track NO SEATBELT alerts for cycling
    
    # Buzzer control: beep on each NO SEATBELT alert (2 times total)
    buzzer_count = 0
    BUZZER_MAX_COUNT = 2

    def add_alert(message: str):
        """
        ULTRA-INSTANT alert printing - NEXT LEVEL optimization
        Multiple layers of instant output for absolute minimum latency
        """
        import sys
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        
        # LEVEL 1: Direct sys.stdout write (bypass print buffering)
        # This is the FASTEST possible console output in Python
        alert_msg = f"[OPTM] [{ts}] [ALERT] {message}\n"
        sys.stdout.write(alert_msg)
        sys.stdout.flush()  # Force immediate flush to terminal
        
        # LEVEL 2: Also write to stderr for redundancy (some terminals prioritize stderr)
        sys.stderr.write(alert_msg)
        sys.stderr.flush()
        
        # LEVEL 3: Queue for delayed overlay (500ms after console)
        # This ensures console prints are seen FIRST
        overlay_queue.append({
            'key': f"{ts} {message}",
            'time': time.time(),
            'display_after': time.time() + OVERLAY_DELAY
        })
        
        # LEVEL 4: Buzzer trigger for NO SEATBELT (up to 3 times)
        nonlocal buzzer_count
        if "NO SEATBELT" in message and buzzer_count < BUZZER_MAX_COUNT:
            buzzer_beep(times=3, on_s=0.07, off_s=0.07)
            buzzer_count += 1
            optm_print("Buzzer", f"NO SEATBELT beep #{buzzer_count}/{BUZZER_MAX_COUNT}")
        
        # Reset buzzer count when seatbelt is worn
        elif "Seatbelt Worn" in message:
            if buzzer_count > 0:
                optm_print("Buzzer", "Seatbelt worn - buzzer reset")
            buzzer_count = 0
        
        return message

    tilt_up_count = 0
    # Treat legacy --show_boxes as alias for --draw_boxes
    draw_boxes = bool(getattr(args, 'draw_boxes', False) or getattr(args, 'show_boxes', False))

    # Initialize KSS (Karolinska Sleepiness Scale) system for AIS 184 compliance
    # NOTE: buzzer_callback=None to prevent alert manager from triggering buzzer
    # We trigger buzzer immediately in override logic instead (bypasses 10s cooldown)
    kss_calculator, kss_alert_manager = create_kss_system(buzzer_callback=None)
    
    # KSS tracking variables
    last_blink_time = 0
    blink_start_time = None
    head_droop_start_time = None
    head_droop_duration = 0.0

    # (MediaPipe caching moved inside MediaPipeWorker)
    # Keep last valid MP results (for async worker gaps)
    last_valid_result = None
    last_valid_hand = None
    last_valid_pose = None

    while cap.isOpened():
        fid += 1
        if _LAT: lat.start_frame(fid)
        ret, frame = cap.read()
        if not ret:
            optm_print("WARNING", "Failed to read frame")
            break
        
        if args.crop_center > 0 and args.crop_center <= 0.5:
            h_orig, w_orig = frame.shape[:2]
            crop_h = int(h_orig * args.crop_center)
            crop_w = int(w_orig * args.crop_center)
            frame = frame[crop_h:h_orig-crop_h, crop_w:w_orig-crop_w]

        h, w = frame.shape[:2]
        smoking_alert_fired_current_frame = False
        mobile_call_candidate_frame = False
        texting_candidate_frame = False
        hand_near_face_any = False
        # Per-frame risk markers (filled by add_alert mapping)
        risk_events = []

        # Optional auto image size tuning
        if getattr(args, 'auto_imgsz', False) and yolo_worker and (fid % 30 == 0):
            # Simple heuristic: 320 for <=720p, 512 for <=1080p, 640 otherwise
            longest = max(h, w)
            target = 320 if longest <= 720 else (512 if longest <= 1080 else 640)
            try:
                if int(getattr(yolo_worker, 'imgsz', target)) != int(target):
                    yolo_worker.imgsz = int(target)
            except Exception:
                pass
        # Submit to YOLO with dynamic skip
        # Adaptive seatbelt skip: run every frame when unstable, relax when stable
        if seatbelt_status != seatbelt_prev_status:
            # Status changed - reset to fast detection
            seatbelt_stable_frames = 0
            seatbelt_prev_status = seatbelt_status
            dynamic_skip = 1  # force every frame
        else:
            # Status unchanged - increment stability counter
            seatbelt_stable_frames += 1
            if seatbelt_stable_frames >= SEATBELT_STABLE_THRESHOLD:
                # Stable for 30+ frames (1 second) - relax to every 15 frames
                dynamic_skip = SEATBELT_RELAXED_SKIP
            else:
                # Still stabilizing - keep checking every frame
                dynamic_skip = 1
        
        # YOLO Cycling: Run every 10 seconds
        current_time = time.time()
        
        # Check if 10 seconds have passed since last YOLO submission
        should_run_yolo = (current_time - last_yolo_submit_time) >= YOLO_INTERVAL
        
        # Submit frame to YOLO if interval elapsed
        if yolo_worker and should_run_yolo:
            yolo_worker.submit(frame)
            last_yolo_submit_time = current_time
            if fid % 30 == 0:  # Log every 30 frames to avoid spam
                optm_print("YOLO", f"Running detection (every {YOLO_INTERVAL}s)")

        dets = last_yolo_dets
        if yolo_worker:
            latest = yolo_worker.get_latest()
            if latest is not None:
                ts, res_dets = latest
                if ts > last_yolo_ts:
                    last_yolo_dets = res_dets
                    last_yolo_ts = ts
                    dets = last_yolo_dets
                    last_yolo_result_time = time.time()

        eye_closed = 0
        head_turn = 0
        hands_free = False
        head_tilt = 0
        head_droop = 0
        yawn = False
        mobile_call_frame = False

        # MediaPipe processing: Simple inline pattern from DMSv8
        # Convert to RGB once for all MediaPipe models
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if _LAT: lat.mark('t_cap')  # Camera capture includes cropping + RGB conversion
        
        # Process face_mesh and hands inline (synchronous)
        result = face_mesh.process(rgb) if face_mesh is not None else None
        hand_result = hands.process(rgb) if hands is not None else None
        pose_result = pose.process(rgb) if pose is not None else None
        if _LAT: lat.mark('t_mp')
        
        # Gaze tracking (process after MediaPipe)
        if gaze_tracker and result and result.multi_face_landmarks:
            gaze_dir, h_ratio, v_ratio = gaze_tracker.process_frame(
                result.multi_face_landmarks[0], w, h
            )
        
        # Debug: MediaPipe status
        if fid % 30 == 0:
            face_status = "Face detected" if result and result.multi_face_landmarks else "No face"
            hand_status = "Hand detected" if hand_result and hand_result.multi_hand_landmarks else "No hand"
            optm_print("Frame", f"{face_status}, {hand_status}", fid)
        if args.no_face_display:
            frame = np.zeros_like(frame)
        current_time = time.time()

        face_present_this_frame = False
        
        # Face lost timer logic (like DMSv8)
        if not (result and getattr(result, 'multi_face_landmarks', None)):
            if 'face_lost_start_time' not in globals() or globals().get('face_lost_start_time') is None:
                globals()['face_lost_start_time'] = current_time
                optm_print("WARNING", "Face lost (likely head turn) - starting timer")
            else:
                face_lost_duration = current_time - globals()['face_lost_start_time']
                if face_lost_duration >= 5:
                    add_alert("Driver Distracted")
                    head_turn = 3
                elif face_lost_duration >= 3:
                    add_alert("Driver Distracted")
                    head_turn = 2
        
        # Accept cached landmarks (do not require face_mesh_ran when using threaded MediaPipe)
        if result and getattr(result, 'multi_face_landmarks', None):
                # Reset face lost timer
                if 'face_lost_start_time' in globals() and globals().get('face_lost_start_time') is not None:
                    optm_print("INFO", "Face re-acquired")
                    globals()['face_lost_start_time'] = None
                landmarks = result.multi_face_landmarks[0].landmark
                last_face_detect_ts = time.time()
                face_present_this_frame = True
                face_center = (int(landmarks[1].x * w), int(landmarks[1].y * h))

                left_ear = get_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = get_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                visible_ears = []
                if left_ear > 0: visible_ears.append(left_ear)
                if right_ear > 0: visible_ears.append(right_ear)
                raw_avg_ear = np.mean(visible_ears) if visible_ears else 1.0
                
                # Apply EMA smoothing to reduce EAR jitter (3D mesh + smoothing)
                if 'smoothed_ear' not in globals():
                    globals()['smoothed_ear'] = raw_avg_ear
                else:
                    SMOOTH_FACTOR = 0.3  # Balanced for responsiveness with 2D gaze
                    globals()['smoothed_ear'] = SMOOTH_FACTOR * raw_avg_ear + (1 - SMOOTH_FACTOR) * globals()['smoothed_ear']
                
                avg_ear = globals()['smoothed_ear']

                visible_iris_points = [
                    idx for idx in LEFT_IRIS + RIGHT_IRIS
                    if 0 <= idx < len(landmarks) and hasattr(landmarks[idx], 'visibility') and landmarks[idx].visibility > 0.1
                ]
                iris_visible = len(visible_iris_points) >= 4

                def iris_center(indices):
                    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
                    return np.mean(pts, axis=0)
                iris_center_avg = (iris_center(LEFT_IRIS) + iris_center(RIGHT_IRIS)) / 2
                iris_y_avg = iris_center_avg[1] / h if h else 0.5
                iris_missing_or_low = (not iris_visible) or (iris_y_avg > 0.5)
                eye_closed_by_ear = avg_ear < args.ear_threshold
                
                # Calculate instantaneous eye openness percentage from EAR
                EAR_FULLY_OPEN = 0.28
                EAR_FULLY_CLOSED = 0.10
                if avg_ear <= EAR_FULLY_CLOSED:
                    current_eye_openness = 0.0
                elif avg_ear >= EAR_FULLY_OPEN:
                    current_eye_openness = 100.0
                else:
                    current_eye_openness = ((avg_ear - EAR_FULLY_CLOSED) / (EAR_FULLY_OPEN - EAR_FULLY_CLOSED)) * 100.0
                
                if fid % 10 == 0:
                    optm_print("Eye Openness", f"{current_eye_openness:.1f}% (EAR: {avg_ear:.3f})")
                
                # Track eye closure duration, speed, and level
                now_eye = time.time()
                
                # Calculate actual frame time delta for accurate speed calculation
                frame_time_delta = now_eye - prev_frame_time
                prev_frame_time = now_eye
                if frame_time_delta <= 0:
                    frame_time_delta = 0.033  # Fallback to ~30fps
                
                # Eye closure speed calculation (rate of EAR change)
                if prev_ear is not None:
                    ear_change = prev_ear - avg_ear  # Positive when closing, negative when opening
                    eye_closure_speed = ear_change / frame_time_delta  # Convert to units per second
                    
                    # Alert on fast closure (speed)
                    if eye_closure_speed > 0.5:  # Rapid closing detected
                        add_alert(f"Eye Closure Speed: RAPID ({eye_closure_speed:.2f}/s)")
                    elif eye_closure_speed > 0.3:
                        add_alert(f"Eye Closure Speed: FAST ({eye_closure_speed:.2f}/s)")
                prev_ear = avg_ear
                
                # Eye closure level based on current openness percentage
                if current_eye_openness <= 20.0:
                    closure_level = "SEVERE (Nearly Closed)"
                elif current_eye_openness <= 40.0:
                    closure_level = "MODERATE (Half Closed)"
                elif current_eye_openness <= 60.0:
                    closure_level = "MILD (Slightly Closed)"
                else:
                    closure_level = "OPEN"
                
                # Eye closure duration tracking (UNIFIED TIMER - single source of truth)
                if eye_closed_by_ear and iris_missing_or_low:
                    if eye_close_start_time is None:
                        eye_close_start_time = now_eye
                    eye_close_duration = now_eye - eye_close_start_time
                    
                    # Alert on duration milestones
                    if eye_close_duration >= 3.0:
                        add_alert(f"Eye Closure Duration: {eye_close_duration:.1f}s - CRITICAL")
                    elif eye_close_duration >= 2.0:
                        add_alert(f"Eye Closure Duration: {eye_close_duration:.1f}s - WARNING")
                    elif eye_close_duration >= 1.0 and int(eye_close_duration * 10) % 5 == 0:
                        add_alert(f"Eye Closure Duration: {eye_close_duration:.1f}s")
                    
                    # Print level when eyes are closed
                    if fid % 5 == 0:
                        add_alert(f"Eye Closure Level: {closure_level} ({current_eye_openness:.1f}%)")
                else:
                    # Eyes opened - reset ALL timers
                    if eye_close_start_time is not None and eye_close_duration > 0.5:
                        add_alert(f"Eyes Reopened (was closed for {eye_close_duration:.1f}s)")
                    eye_close_start_time = None
                    eye_close_duration = 0.0
                    eye_closure_latency_logged = False  # Reset flag for next episode
                    buzzer_triggered = False  # Reset buzzer flag on eye reopen

                if eye_closed_by_ear and iris_missing_or_low:
                    eye_closure_counter += 1
                    if eye_closure_counter > 30:
                        add_alert("Alert: Eyes Closed Too Long"); eye_closed = 2
                    elif eye_closure_counter > args.eye_closed_frames_threshold:
                        add_alert("Warning: Eyes Closed"); eye_closed = 1
                else:
                    if 2 <= eye_closure_counter < args.eye_closed_frames_threshold:
                        blink_counter += 1
                    eye_closure_counter = 0

                curr_eye_state = 'closed' if (eye_closed_by_ear and iris_missing_or_low) else 'open'
                if prev_eye_state is None:
                    prev_eye_state = curr_eye_state
                elif curr_eye_state != prev_eye_state:
                    add_alert("Eye Closed" if curr_eye_state == 'closed' else "Eye Open")
                    # Count blinks on transition OPEN -> CLOSED
                    try:
                        if (prev_eye_state == 'open') and (curr_eye_state == 'closed'):
                            blink_total += 1
                            blink_start_time = current_time
                        elif (prev_eye_state == 'closed') and (curr_eye_state == 'open'):
                            # Blink completed - record duration for KSS
                            if blink_start_time is not None:
                                blink_duration = current_time - blink_start_time
                                if 0.05 < blink_duration < 1.0:  # Valid blink range
                                    kss_calculator.add_blink(blink_duration, current_time)
                                blink_start_time = None
                    except Exception:
                        pass
                    prev_eye_state = curr_eye_state

                if current_time - blink_timer > 60:
                    if blink_counter >= args.blink_rate_threshold:
                        add_alert("High Blinking Rate")
                    blink_counter = 0
                    blink_timer = current_time

                mar = get_mar(landmarks, MOUTH, w, h)
                mar_deque.append(mar)
                if mar > args.mar_threshold:
                    yawn_counter += 1
                if yawn_counter > args.yawn_threshold:
                    add_alert("Warning: Yawning")
                    yawn = True
                    yawn_counter = 0
                    # Record yawn for KSS calculator
                    kss_calculator.add_yawn(current_time)
                    # Trigger buzzer - short single beep for yawning
                    buzzer_beep(times=1, on_s=0.1, off_s=0.0)

                # Simple 2D gaze calculation (from reference file - fast and accurate)
                def iris_center_func(indices):
                    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
                    return np.mean(pts, axis=0)
                
                left_iris_pos = iris_center_func(LEFT_IRIS)
                right_iris_pos = iris_center_func(RIGHT_IRIS)
                iris_center_avg = (left_iris_pos + right_iris_pos) / 2
                gaze_x_norm = float(iris_center_avg[0] / w) if w else 0.5
                
                # Apply smoothing to reduce jitter
                raw_head_x = landmarks[1].x
                raw_head_y = landmarks[1].y
                
                if 'smoothed_head_x' not in globals():
                    globals()['smoothed_head_x'] = raw_head_x
                    globals()['smoothed_head_y'] = raw_head_y
                else:
                    SMOOTH_FACTOR = 0.3  # Balanced for responsiveness with 2D gaze
                    globals()['smoothed_head_x'] = SMOOTH_FACTOR * raw_head_x + (1 - SMOOTH_FACTOR) * globals()['smoothed_head_x']
                    globals()['smoothed_head_y'] = SMOOTH_FACTOR * raw_head_y + (1 - SMOOTH_FACTOR) * globals()['smoothed_head_y']
                
                head_x = globals()['smoothed_head_x']
                head_y = globals()['smoothed_head_y']

                if calibration_mode:
                    # cv2.putText(frame, "AUTO-CALIBRATING: Just look at road naturally!", (10, h - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    pass
                else:
                    gaze_offset = abs(gaze_x_norm - gaze_center)
                    dx = (head_x - head_center_x)
                    dy = (head_y - head_center_y)
                    alpha = 0.3
                    sm_head_x_signed = alpha * dx + (1 - alpha) * sm_head_x_signed
                    sm_head_y_signed = alpha * dy + (1 - alpha) * sm_head_y_signed
                    sm_head_x_offset = abs(sm_head_x_signed)
                    sm_head_y_offset = abs(sm_head_y_signed)

                    # DISABLED: Adaptive recalibration interferes with gaze detection
                    # Only recalibrate head position, NOT gaze_center
                    eyes_open_forward = (abs(sm_head_x_signed) < 0.05) and (abs(sm_head_y_signed) < 0.05) and (prev_eye_state != 'closed')
                    if eyes_open_forward:
                        beta = 0.05
                        head_center_x = (1 - beta) * head_center_x + beta * head_x
                        head_center_y = (1 - beta) * head_center_y + beta * head_y
                        # gaze_center NOT updated - keeps calibrated baseline

                    # Simple 2D gaze detection (best approach for side-mounted cameras)
                    gaze_deviation = gaze_x_norm - gaze_center  # negative=left, positive=right
                    
                    # Debug less frequently to reduce noise
                    if fid % 30 == 0:
                        optm_print("GAZE", f"deviation={gaze_deviation:.3f}, center={gaze_center:.3f}, L_streak={gaze_left_confirm_streak}, R_streak={gaze_right_confirm_streak}")
                    
                    now_gaze = time.time()
                    
                    # PRODUCTION THRESHOLDS: Balanced for accuracy
                    LEFT_THRESHOLD = 0.025   # 2.5% deviation (reduced false positives)
                    RIGHT_THRESHOLD = 0.025  # 2.5% deviation (symmetric for fairness)
                    UP_THRESHOLD = 0.020     # 2.0% vertical deviation (looking up - mirrors)
                    DOWN_THRESHOLD = 0.025   # 2.5% vertical deviation (looking down - phone/dashboard)
                    
                    # FILTER: Suppress gaze detection during hand activities (smoking/phone/texting)
                    # This prevents false positives when hand is near face
                    hand_activity_suppression = False
                    try:
                        # Check if variables exist from hand detection (defined later in code)
                        if 'hand_near_mouth_now' in locals() and hand_near_mouth_now:
                            hand_activity_suppression = True
                        if 'mobile_call_candidate_frame' in locals() and mobile_call_candidate_frame:
                            hand_activity_suppression = True
                        if 'texting_hand_detected' in locals() and texting_hand_detected:
                            hand_activity_suppression = True
                    except:
                        pass
                    
                    # Reset gaze streaks if hand activity detected
                    if hand_activity_suppression:
                        gaze_left_confirm_streak = 0
                        gaze_right_confirm_streak = 0
                        gaze_alerted_left = False
                        gaze_alerted_right = False
                    else:
                        # HORIZONTAL GAZE DETECTION (Left/Right mirror checks)
                        # Detect LEFT gaze (negative deviation)
                        if gaze_deviation < -LEFT_THRESHOLD:
                            gaze_left_confirm_streak += 1
                            gaze_right_confirm_streak = 0
                            
                            if gaze_left_confirm_streak >= GAZE_CONFIRM_FRAMES:
                                if not gaze_alerted_left and (now_gaze - last_gaze_left_alert_time >= GAZE_ALERT_COOLDOWN):
                                    add_alert("Looking Right")
                                    dms_print("ALERT", f"Right GAZE TRIGGERED! (deviation={gaze_deviation:.3f})")
                                    gaze_alerted_left = True
                                    last_gaze_left_alert_time = now_gaze
                        
                        # Detect RIGHT gaze (positive deviation)
                        elif gaze_deviation > RIGHT_THRESHOLD:
                            gaze_right_confirm_streak += 1
                            gaze_left_confirm_streak = 0
                            
                            if gaze_right_confirm_streak >= GAZE_CONFIRM_FRAMES:
                                if not gaze_alerted_right and (now_gaze - last_gaze_right_alert_time >= GAZE_ALERT_COOLDOWN):
                                    add_alert("Looking Left")
                                    dms_print("ALERT", f"Left GAZE TRIGGERED! (deviation={gaze_deviation:.3f})")
                                    gaze_alerted_right = True
                                    last_gaze_right_alert_time = now_gaze
                        
                        # Reset when gaze returns to center
                        else:
                            if gaze_left_confirm_streak > 0 or gaze_right_confirm_streak > 0:
                                gaze_left_confirm_streak = 0
                                gaze_right_confirm_streak = 0
                                gaze_alerted_left = False
                                gaze_alerted_right = False
                    
                    # VERTICAL GAZE DETECTION (Up/Down for mirrors/phone)
                    # Calculate vertical iris position
                    try:
                        iris_y_norm = float(iris_center_avg[1] / h) if h > 0 else 0.5
                        
                        # Initialize vertical gaze center if not exists
                        if 'iris_center_y' not in globals():
                            globals()['iris_center_y'] = iris_y_norm
                        
                        # Calculate vertical deviation
                        vertical_deviation = iris_y_norm - globals()['iris_center_y']
                        
                        # Initialize vertical gaze tracking variables
                        if 'gaze_up_confirm_streak' not in globals():
                            globals()['gaze_up_confirm_streak'] = 0
                            globals()['gaze_down_confirm_streak'] = 0
                            globals()['gaze_alerted_up'] = False
                            globals()['gaze_alerted_down'] = False
                            globals()['last_gaze_up_alert_time'] = 0
                            globals()['last_gaze_down_alert_time'] = 0
                        
                        # Skip vertical detection during hand activity
                        if not hand_activity_suppression:
                            # Looking UP (negative deviation - rear mirror check)
                            if vertical_deviation < -UP_THRESHOLD:
                                globals()['gaze_up_confirm_streak'] += 1
                                globals()['gaze_down_confirm_streak'] = 0
                                
                                if globals()['gaze_up_confirm_streak'] >= GAZE_CONFIRM_FRAMES:
                                    if not globals()['gaze_alerted_up'] and (now_gaze - globals()['last_gaze_up_alert_time'] >= GAZE_ALERT_COOLDOWN):
                                        add_alert("Looking Up - Mirror Check")
                                        globals()['gaze_alerted_up'] = True
                                        globals()['last_gaze_up_alert_time'] = now_gaze
                            
                            # Looking DOWN (positive deviation - phone/dashboard)
                            elif vertical_deviation > DOWN_THRESHOLD:
                                globals()['gaze_down_confirm_streak'] += 1
                                globals()['gaze_up_confirm_streak'] = 0
                                
                                if globals()['gaze_down_confirm_streak'] >= GAZE_CONFIRM_FRAMES:
                                    if not globals()['gaze_alerted_down'] and (now_gaze - globals()['last_gaze_down_alert_time'] >= GAZE_ALERT_COOLDOWN):
                                        add_alert("Looking Down - Phone/Dashboard")
                                        globals()['gaze_alerted_down'] = True
                                        globals()['last_gaze_down_alert_time'] = now_gaze
                            
                            # Reset when gaze returns to center vertically
                            else:
                                if globals()['gaze_up_confirm_streak'] > 0 or globals()['gaze_down_confirm_streak'] > 0:
                                    globals()['gaze_up_confirm_streak'] = 0
                                    globals()['gaze_down_confirm_streak'] = 0
                                    globals()['gaze_alerted_up'] = False
                                    globals()['gaze_alerted_down'] = False
                        else:
                            # Reset vertical gaze during hand activity
                            globals()['gaze_up_confirm_streak'] = 0
                            globals()['gaze_down_confirm_streak'] = 0
                            globals()['gaze_alerted_up'] = False
                            globals()['gaze_alerted_down'] = False
                    except Exception:
                        pass  # Silently skip vertical gaze if calculation fails

                    ang = compute_head_angles(landmarks, w, h)
                    yaw_s = pitch_s = None
                    if ang is not None:
                        yaw_deg, pitch_deg, _ = ang
                        if not hasattr(main, '_yaw_s'): main._yaw_s = yaw_deg
                        if not hasattr(main, '_pitch_s'): main._pitch_s = pitch_deg
                        ema = 0.3
                        main._yaw_s = (1 - ema) * main._yaw_s + ema * yaw_deg
                        main._pitch_s = (1 - ema) * main._pitch_s + ema * pitch_deg
                        yaw_s = main._yaw_s; pitch_s = main._pitch_s

                    # Check upward/downward FIRST (higher priority)
                    yoff = abs(sm_head_y_offset)
                    
                    # DEBUG: Print head position every 30 frames to see actual values
                    if fid % 30 == 0:
                        optm_print("HeadPos", f"Y_signed={sm_head_y_signed:.3f}, Y_offset={yoff:.3f}")
                    
                    # CORRECT MAPPING (matching DMSv8 and image coordinates):
                    # sm_head_y_signed > 0 → head_y > head_center_y → Y increased → HEAD MOVED DOWN in frame → Looking Downward/Drooping
                    # sm_head_y_signed < 0 → head_y < head_center_y → Y decreased → HEAD MOVED UP in frame → Looking Upward
                    
                    if sm_head_y_signed < 0:
                        # Looking UPWARD - Y coordinate decreased (head moved up in frame)
                        if yoff >= 0.015:  # Mild threshold logic
                            head_tilt = 1
                            add_alert("Looking Upward")
                    elif sm_head_y_signed > 0:
                        # HEAD DROOPING/DOWNWARD - Y coordinate increased (head moved down in frame)
                        if yoff >= 0.05:  # Mild threshold logic
                            head_droop = 1
                            add_alert("Head Downward")

                    # Then check horizontal turns (only if no vertical detected)
                    if head_tilt == 0 and head_droop == 0 and yaw_s is not None:
                        # Asymmetric thresholds for side-mounted camera (45 degree angle)
                        direction = 'Right' if ((yaw_s >= 0) ^ bool(args.flip_yaw_sign)) else 'Left'
                        ay = abs(yaw_s)
                        
                        # Different thresholds for left vs right
                        if direction == 'Left' and ay >= 50:  # Left turn: 50 degrees
                            add_alert(f"Head Turn Left")
                            head_turn = 1
                        elif direction == 'Right' and ay >= 15:  # Right turn: 15 degrees (more sensitive)
                            add_alert(f"Head Turn Right")
                            head_turn = 1

                # Draw face mesh
                if (not args.no_mesh_display) and (mp_drawing is not None):
                    mp_drawing.draw_landmarks(
                        frame, result.multi_face_landmarks[0], face_mesh_connections,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

                eyes_apparently_open = not (eye_closed_by_ear and iris_missing_or_low)
                # Draw gaze and iris
                if eyes_apparently_open:
                    try:
                        left_center, left_r = get_iris_center_and_radius(landmarks, LEFT_IRIS, w, h)
                        right_center, right_r = get_iris_center_and_radius(landmarks, RIGHT_IRIS, w, h)
                        gaze_vec, _ = estimate_gaze_direction(landmarks, w, h)
                        if gaze_vec is not None:
                            def draw_line(center, r):
                                vx, vy = gaze_vec
                                length = float(r) * 3.0
                                x0, y0 = int(center[0]), int(center[1])
                                x1, y1 = int(x0 + vx * length), int(y0 + vy * length)
                                if getattr(args, 'retina_overlay', False):
                                    cv2.line(frame, (x0, y0), (x1, y1), (220, 220, 220), 1)
                                else:
                                    # Unified non-red/non-yellow color: sky blue
                                    cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                            draw_line(left_center, left_r)
                            draw_line(right_center, right_r)
                        cv2.circle(frame, left_center, 2, (0, 0, 255), -1)
                        cv2.circle(frame, right_center, 2, (0, 0, 255), -1)
                    except Exception:
                        pass

        # Initialize hand-related variables for smoking/phone detection
        hand_near_mouth_now = False
        hand_at_ear_and_face = False
        
        # Initialize mouth-related variables (needed for hand-to-mouth detection)
        have_mouth = False
        mouth_cx = mouth_cy = 0
        face_h_for_mouth = 0
        if result and getattr(result, 'multi_face_landmarks', None):
            try:
                lms = result.multi_face_landmarks[0].landmark
                mx = np.mean([lms[i].x for i in MOUTH])
                my = np.mean([lms[i].y for i in MOUTH])
                mouth_cx = int(mx * w)
                mouth_cy = int(my * h)
                if face_bbox is not None:
                    face_h_for_mouth = max(1, face_bbox[3] - face_bbox[1])
                else:
                    face_h_for_mouth = max(1, int(0.35 * h))
                have_mouth = True
            except Exception:
                have_mouth = False

        # Hands proximity & texting (+ optional medical heuristics)
        if (hand_result is not None) and getattr(hand_result, 'multi_hand_landmarks', None) and result and getattr(result, 'multi_face_landmarks', None):
            landmarks = result.multi_face_landmarks[0].landmark
            face_center = (int(landmarks[1].x * w), int(landmarks[1].y * h))
            # Build a chest ROI using pose landmarks if available, else derive from face bbox
            chest_roi_local = None
            if (pose_result is not None) and getattr(pose_result, 'pose_landmarks', None):
                try:
                    plms = pose_result.pose_landmarks.landmark
                    l_sh = plms[LEFT_SHOULDER]; r_sh = plms[RIGHT_SHOULDER]
                    l_hp = plms[LEFT_HIP]; r_hp = plms[RIGHT_HIP]
                    def vis_ok(lm):
                        return hasattr(lm, 'visibility') and lm.visibility is not None and lm.visibility > 0.45
                    if vis_ok(l_sh) and vis_ok(r_sh):
                        lx, ly = int(l_sh.x * w), int(l_sh.y * h)
                        rx, ry = int(r_sh.x * w), int(r_sh.y * h)
                        sh_width = max(1, abs(rx - lx))
                        sh_y = (ly + ry) // 2
                        cx1 = max(0, min(lx, rx) - int(0.15 * sh_width))
                        cx2 = min(w - 1, max(lx, rx) + int(0.15 * sh_width))
                        if vis_ok(l_hp) and vis_ok(r_hp):
                            hy = (int(l_hp.y * h) + int(r_hp.y * h)) // 2
                            torso_h = max(1, hy - sh_y)
                            cy1 = max(0, sh_y - int(0.10 * torso_h))
                            cy2 = min(h - 1, sh_y + int(0.70 * torso_h))
                        else:
                            cy1 = max(0, sh_y - int(0.05 * h))
                            cy2 = min(h - 1, sh_y + int(0.25 * h))
                        if cy2 > cy1 and cx2 > cx1:
                            chest_roi_local = (cx1, cy1, cx2, cy2)
                except Exception:
                    chest_roi_local = None
            if chest_roi_local is None:
                # Fallback from face bbox
                try:
                    if face_bbox is not None:
                        fx1, fy1, fx2, fy2 = face_bbox
                        fw = max(1, fx2 - fx1)
                        fh = max(1, fy2 - fy1)
                        cy1 = max(0, int(fy2 + 0.15 * fh))
                        cy2 = min(h - 1, int(min(h - 1, fy2 + 2.0 * fh)))
                        cx1 = max(0, int(max(0, fx1 - 0.5 * fw)))
                        cx2 = min(w - 1, int(min(w - 1, fx2 + 0.5 * fw)))
                        chest_roi_local = (cx1, cy1, cx2, cy2) if cy2 > cy1 and cx2 > cx1 else None
                except Exception:
                    chest_roi_local = None
            # Optionally expand the chest ROI a bit
            if chest_roi_local is not None:
                try:
                    exp = float(getattr(args, 'chest_roi_expand', 0.10) or 0.0)
                except Exception:
                    exp = 0.10
                if exp > 0:
                    x1, y1, x2, y2 = chest_roi_local
                    cw = x2 - x1; ch = y2 - y1
                    dx = int(exp * cw); dy = int(exp * ch)
                    x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
                    x2 = min(w - 1, x2 + dx); y2 = min(h - 1, y2 + dy)
                    chest_roi_local = (x1, y1, x2, y2)
            coords = []
            hand_centers_px = []
            
            for idx, hlm in enumerate(hand_result.multi_hand_landmarks):
                # Check if hand is near ear (phone call indicator)
                near_ear = hand_near_ear(landmarks, hlm, w, h)
                # Check if hand is near face center (general proximity)
                near_face = hand_near_face(face_center, hlm, frame.shape, px=args.hand_near_face_px)
                
                # NEW LOGIC: Mobile call = hand near BOTH ear AND face simultaneously
                if near_ear and near_face:
                    hand_at_ear_and_face = True
                    mobile_call_candidate_frame = True
                
                # NEW LOGIC: Check if hand is near mouth area (for smoking detection)
                if have_mouth:
                    # Calculate distance from hand center to mouth center
                    hand_xs = [lm.x for lm in hlm.landmark]
                    hand_ys = [lm.y for lm in hlm.landmark]
                    hand_x = np.mean(hand_xs) * w
                    hand_y = np.mean(hand_ys) * h
                    
                    # Mouth area proximity threshold (normalize by face height)
                    mouth_proximity_radius = face_h_for_mouth * 0.35 if face_h_for_mouth > 0 else 80
                    dist_to_mouth = np.hypot(hand_x - mouth_cx, hand_y - mouth_cy)
                    
                    if dist_to_mouth < mouth_proximity_radius:
                        hand_near_mouth_now = True
                
                if not args.no_mesh_display and mp_drawing is not None and hand_connections is not None:
                    mp_drawing.draw_landmarks(frame, hlm, hand_connections)
                xs = [lm.x for lm in hlm.landmark]; ys = [lm.y for lm in hlm.landmark]
                mxn, myn = (np.mean(xs), np.mean(ys))
                coords.append((mxn, myn))
                hand_centers_px.append((int(mxn * w), int(myn * h)))

            # Enhanced phone detection logic
            if not calibration_mode:
                # Case 1: Two hands together (texting - original logic)
                if len(coords) == 2:
                    (x1, y1), (x2, y2) = coords
                    dist = np.hypot(x2 - x1, y2 - y1)
                    both_low = y1 > 0.6 and y2 > 0.6
                    # Require not near either ear
                    not_near_ear = not hand_near_ear(landmarks, hand_result.multi_hand_landmarks[0], w, h) and \
                                   not hand_near_ear(landmarks, hand_result.multi_hand_landmarks[1], w, h)
                    if dist < 0.35 and both_low and not_near_ear:
                        texting_candidate_frame = True
            
            # NEW: Simple hand-to-mouth counter for smoking detection
            
            # Detect state transitions: hand moved TO mouth or AWAY from mouth
            if hand_near_mouth_now and not hand_near_mouth_state:
                # Hand just moved TO mouth → increment counter
                hand_mouth_counter += 1
                hand_near_mouth_state = True
            elif not hand_near_mouth_now and hand_near_mouth_state:
                # Hand moved AWAY from mouth
                hand_near_mouth_state = False

            if mobile_call_candidate_frame or texting_candidate_frame or hand_near_face_any:
                hands_free = True

            # Single-handed texting detection: hand in chest ROI (not stationary)
            texting_hand_detected = False
            if chest_roi_local is not None and len(hand_centers_px) > 0:
                x1c, y1c, x2c, y2c = chest_roi_local
                # Check if any hand is in chest ROI (not requiring stationary)
                for idx, (hc_x, hc_y) in enumerate(hand_centers_px):
                    in_chest = (x1c <= hc_x <= x2c) and (y1c <= hc_y <= y2c)
                    # Exclude if it's a phone call or smoking
                    if in_chest and not mobile_call_candidate_frame and smoke_status not in ('mouth', 'hand'):
                        texting_hand_detected = True
                        break
            
            # Track texting confirmation
            now_texting = time.time()
            if texting_hand_detected:
                texting_hand_confirm_streak += 1
                if texting_hand_confirm_streak >= TEXTING_HAND_CONFIRM_FRAMES:
                    if not texting_hand_alerted and (now_texting - last_texting_hand_alert_time >= TEXTING_HAND_ALERT_COOLDOWN):
                        add_alert("Single-Handed Texting Observed")
                        texting_hand_alerted = True
                        last_texting_hand_alert_time = now_texting
            else:
                texting_hand_confirm_streak = 0
                texting_hand_alerted = False

            # Optional medical distress: chest clutching detection
            if medical_enabled and chest_roi_local is not None:
                now_med = time.time()
                # Respect cooldown after an episode ends
                if last_chest_clutch_end_ts and (now_med - last_chest_clutch_end_ts) < float(getattr(args, 'medical_cooldown_secs', 6.0) or 6.0):
                    pass  # skip detection during cooldown
                else:
                    x1c, y1c, x2c, y2c = chest_roi_local
                    # Determine if any hand center lies within the chest ROI and is stationary
                    any_on_chest_and_stationary = False
                    # Compute motion threshold in pixels per frame approximated by per-second epsilon
                    try:
                        eps_norm_per_s = float(getattr(args, 'chest_stationary_eps', 0.03) or 0.03)
                    except Exception:
                        eps_norm_per_s = 0.03
                    # Estimate dt ~ 1/30 if available
                    dt_frame = 1.0 / 30.0
                    px_eps = eps_norm_per_s * h * dt_frame
                    for idx, (hc_x, hc_y) in enumerate(hand_centers_px):
                        in_roi = (x1c <= hc_x <= x2c) and (y1c <= hc_y <= y2c)
                        if not in_roi:
                            continue
                        # Stationary check vs previous frame position for the same hand index
                        try:
                            if idx < len(prev_hand_centers_px):
                                px_prev = prev_hand_centers_px[idx]
                                if px_prev is not None:
                                    dist = math.hypot(hc_x - px_prev[0], hc_y - px_prev[1])
                                else:
                                    dist = 0.0
                            else:
                                dist = 0.0
                        except Exception:
                            dist = 0.0
                        # Also suppress if near ear/face candidates active this frame, or active smoking
                        suppress = mobile_call_candidate_frame or hand_near_face_any or (smoke_status in ('mouth','hand'))
                        if in_roi and (dist <= px_eps) and (not suppress):
                            any_on_chest_and_stationary = True
                            break

                    if any_on_chest_and_stationary:
                        if chest_clutch_started_ts == 0.0:
                            chest_clutch_started_ts = now_med
                        # Trigger when duration satisfied
                        min_secs = float(getattr(args, 'chest_clutch_min_secs', 4.0) or 4.0)
                        if (now_med - chest_clutch_started_ts) >= min_secs:
                            if not chest_clutch_active:
                                chest_clutch_active = True
                                last_medical_emit = now_med
                                add_alert("Medical: Chest clutching")
                            elif (now_med - last_medical_emit) >= float(getattr(args, 'medical_emit_interval', 2.0) or 2.0):
                                last_medical_emit = now_med
                                add_alert("Medical: Chest clutching")
                    else:
                        # reset/cooldown when condition breaks
                        if chest_clutch_active:
                            last_chest_clutch_end_ts = now_med
                        chest_clutch_active = False
                        chest_clutch_started_ts = 0.0

            # Update previous hand centers after medical checks
            prev_hand_centers_px = hand_centers_px

        # Track head droop duration (start timer as soon as head_droop detected)
        if head_droop >= 1 and head_turn == 0:  # Changed from >= 2 to >= 1
            droop_active_since = droop_active_since or time.time()
        else:
            droop_active_since = None
        
        # Track head turn duration
        if head_turn >= 1 or head_tilt >= 1:
            head_turn_active_since = head_turn_active_since or time.time()
        else:
            head_turn_active_since = None

        # Collect calibration samples during countdown
        if calibration_mode and result and getattr(result, 'multi_face_landmarks', None):
            lms = result.multi_face_landmarks[0].landmark
            gaze_x_norm = ((np.array([[lms[i].x * w, lms[i].y * h] for i in LEFT_IRIS]).mean(axis=0) +
                           np.array([[lms[i].x * w, lms[i].y * h] for i in RIGHT_IRIS]).mean(axis=0)) / 2)[0] / w
            calib_gaze_x_samples.append(float(gaze_x_norm))
            calib_head_x_samples.append(float(lms[1].x))
            calib_head_y_samples.append(float(lms[1].y))
        
        if calibration_mode:
            countdown = int(calibration_duration - (time.time() - calibration_start_time))
            if countdown < 0:
                countdown = 0
            if countdown > 0:
                if countdown % 5 == 0 and fid % 30 == 0:
                    # if result and getattr(result, 'multi_face_landmarks', None):
                    #     dms_print("Calibration", f"{countdown}s remaining - Face detected OK (samples: {len(calib_gaze_x_samples)})")
                    # else:
                    #     dms_print("Calibration", f"{countdown}s remaining - WARNING: No face detected!")
                    pass
                # Show countdown only
                cv2.putText(frame, f"{countdown}s", (20, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            if countdown == 0 and calibration_mode:
                # Use median of collected samples for robust baseline
                if len(calib_gaze_x_samples) >= calib_min_samples:
                    gaze_center = float(np.median(calib_gaze_x_samples))
                    head_center_x = float(np.median(calib_head_x_samples))
                    head_center_y = float(np.median(calib_head_y_samples))
                    calibration_mode = False
                    dms_print("Calibration", f"COMPLETE! Gaze center LOCKED at {gaze_center:.3f}")
                    dms_print("Calibration", f"Samples: {len(calib_gaze_x_samples)}, Head center: ({head_center_x:.3f}, {head_center_y:.3f})")
                elif result and getattr(result, 'multi_face_landmarks', None):
                    # Fallback to single-frame if insufficient samples
                    lms = result.multi_face_landmarks[0].landmark
                    gaze_center = float(((np.array([[lms[i].x * w, lms[i].y * h] for i in LEFT_IRIS]).mean(axis=0) +
                                          np.array([[lms[i].x * w, lms[i].y * h] for i in RIGHT_IRIS]).mean(axis=0)) / 2)[0] / w)
                    head_center_x = lms[1].x
                    head_center_y = lms[1].y
                    calibration_mode = False
                    dms_print("Calibration", f"COMPLETE (fallback)! Gaze center LOCKED at {gaze_center:.3f}")
                    add_alert("Calibration Complete")
                else:
                    calibration_mode = False
                    dms_print("Calibration", "WARNING - Completed without face detection! Using defaults.")

        # Severity combos
        now_t = time.time()
        
        # Track head droop duration for KSS (removed "Drowsiness Observed" alert/buzzer)
        if (head_droop >= 1) and (eye_closed >= 1):
            if (droop_active_since is not None) and ((now_t - droop_active_since) >= DROOP_SUSTAIN_SECONDS):
                if head_droop_start_time is None:
                    head_droop_start_time = now_t
                head_droop_duration = now_t - head_droop_start_time
        else:
            # Reset head droop tracking when alert clears
            if head_droop_start_time is not None:
                head_droop_start_time = None
                head_droop_duration = 0.0
        
        # Removed "Driver Distracted" alert and buzzer
        
        # Calculate KSS score (1-9) for AIS 184 compliance
        try:
            # CRITICAL FIX: Calculate eye_close_duration HERE using current timestamp
            # This ensures we use the LATEST value, not stale data from face detection block
            now_t = time.time()
            if eye_close_start_time is not None:
                eye_close_duration = now_t - eye_close_start_time
            else:
                eye_close_duration = 0.0
            
            kss_features = {
                'ear_current': avg_ear if 'avg_ear' in locals() else 0.28,
                'head_droop_duration': head_droop_duration if 'head_droop_duration' in locals() else 0.0,
                'head_droop_active': (head_droop >= 1) and (eye_closed >= 1) if 'head_droop' in locals() and 'eye_closed' in locals() else False
            }
            
            # DEBUG: Print eye closure status every 30 frames
            if fid % 30 == 0 and eye_close_start_time is not None:
                print(f"[EyeClosureDebug] Frame {fid}: duration={eye_close_duration:.2f}s, threshold=1.5s, will_trigger={eye_close_duration > 1.5}")
            
            # Normal KSS calculation (will be overridden if needed)
            kss_score, kss_confidence = kss_calculator.calculate_kss_score(kss_features)
            
            # Calculate head droop duration for override logic
            droop_duration_direct = 0.0
            if droop_active_since is not None:
                droop_duration_direct = now_t - droop_active_since
            
            # ========== CRITICAL OVERRIDE CHECKS ==========
            override_triggered = False
            
            # PRIORITY 1: Eyes closed >1.5s → KSS 9.0 (MOST DANGEROUS)
            if eye_close_duration > 1.5:
                if _LAT: lat.mark('t_kss')  # Mark BEFORE override (correct timing)
                kss_score = 9.0
                kss_confidence = 0.95
                override_triggered = True
                optm_print("KSS Override", f"🚨 CRITICAL: Eyes closed {eye_close_duration:.1f}s → KSS=9.0 (DANGER)")
                
                # IMMEDIATE BUZZER (bypass alert manager cooldown) - ONCE ONLY
                # Optimized: 5 fast beeps = 0.5s total (was 8 beeps = 1.3s)
                if not buzzer_triggered:
                    buzzer_beep(times=5, on_s=0.05, off_s=0.05)
                    buzzer_triggered = True
                    print(f"[Buzzer] IMMEDIATE TRIGGER - Eyes closed {eye_close_duration:.2f}s")
            
            # PRIORITY 2: Head droop overrides (only if eyes not already critical)
            elif droop_duration_direct > 1.0:
                if _LAT: lat.mark('t_kss')  # Mark BEFORE override
                
                # Rule 1: Quick head nod (1-2 seconds) → KSS 7.0
                if droop_duration_direct <= 2.0:
                    kss_score = 7.0  # Warning - single head nod detected
                    kss_confidence = 0.80
                    override_triggered = True
                    optm_print("KSS Override", f"WARNING: Head nod {droop_duration_direct:.1f}s → KSS=7.0")
                # Rule 2: Sustained head droop >2 seconds
                else:
                    # Path 1: Eyes closed + sustained head droop → KSS 9.0
                    if avg_ear < 0.18:
                        kss_score = 9.0  # Maximum severity - eyes closed + head droop
                        kss_confidence = 0.95
                        override_triggered = True
                        optm_print("KSS Override", f"CRITICAL: Eyes closed + Head droop {droop_duration_direct:.1f}s → KSS=9.0")
                    # Path 2: Sustained head droop alone (eyes may be open) → KSS 8.0
                    else:
                        kss_score = 8.0  # Critical - head droop alone
                        kss_confidence = 0.85
                        override_triggered = True
                        optm_print("KSS Override", f"CRITICAL: Head droop {droop_duration_direct:.1f}s → KSS=8.0")
            
            # If no override, mark t_kss after normal calculation
            if not override_triggered and _LAT:
                lat.mark('t_kss')
            
            # KSS >= 7.0 threshold tracking
            if _LAT and kss_score >= 7.0 and not kss7_latency_logged:
                lat.mark('t_buzzer')
                kss7_latency_logged = True
            
            # Reset KSS >= 7.0 flag when score drops below threshold
            if kss_score < 7.0 and kss7_latency_logged:
                kss7_latency_logged = False
            
            # Check for AIS 184 alerts (KSS >= 7)
            # Mark buzzer timing BEFORE alert (captures trigger time, not beep duration)
            if _LAT and override_triggered and eye_close_duration > 1.5 and not eye_closure_latency_logged:
                lat.mark('t_buzzer')
            
            kss_alert = kss_alert_manager.check_and_trigger_alert(kss_score, kss_confidence, now_t)
            if kss_alert:
                add_alert(f"[AIS 184] {kss_alert['message']} (KSS={kss_score:.1f})")
                
                # Log latency for eye closure >1.5s (writes to FILE)
                if _LAT and override_triggered and eye_close_duration > 1.5 and not eye_closure_latency_logged:
                    lat.log_kss_pipeline(fid)
                    eye_closure_latency_logged = True
            
            # Print KSS to console (clean format)
            kss_label = kss_calculator.get_kss_label(kss_score)
            optm_print("KSS", f"Score: {kss_score:.1f}/9 - {kss_label}")
        except Exception as e:
            kss_score = 3.0
            kss_confidence = 0.0

    # YOLO-driven alerts
        if dets:
            def rect_iou(a, b):
                ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0:
                    return 0.0
                area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                denom = float(area_a + area_b - inter)
                return (inter / denom) if denom > 0 else 0.0
            no_seatbelt_hit = False
            seatbelt_worn_hit = False
            smoking_hit = False
            smoking_mouth_hit = False
            smoking_hand_hit = False
            nocig_hit = False
            # Track any raw cigarette_* detection regardless of gating to avoid contradictory 'No Cigarette'
            raw_cig_present = False
            # Collect YOLO class names for optional raw overlay
            yolo_names_this_frame = []
            yolo_eye_open_hit = False
            yolo_eye_closed_hit = False
            no_seatbelt_conf_max = 0.0
            seatbelt_worn_conf_max = 0.0
            torso_roi = None

            if pose_result is not None and getattr(pose_result, 'pose_landmarks', None):
                try:
                    plms = pose_result.pose_landmarks.landmark
                    l_sh = plms[LEFT_SHOULDER]
                    r_sh = plms[RIGHT_SHOULDER]
                    l_hp = plms[LEFT_HIP]
                    r_hp = plms[RIGHT_HIP]
                    def vis_ok(lm):
                        return hasattr(lm, 'visibility') and lm.visibility is not None and lm.visibility > 0.45
                    shoulders_ok = vis_ok(l_sh) and vis_ok(r_sh)
                    hips_ok = vis_ok(l_hp) and vis_ok(r_hp)
                    if shoulders_ok:
                        lx, ly = int(l_sh.x * w), int(l_sh.y * h)
                        rx, ry = int(r_sh.x * w), int(r_sh.y * h)
                        sh_width = max(1, abs(rx - lx))
                        sh_y = (ly + ry) // 2
                        def clip(v, lo, hi):
                            return max(lo, min(hi, v))
                        tx1 = clip(min(lx, rx) - int(0.15 * sh_width), 0, w - 1)
                        tx2 = clip(max(lx, rx) + int(0.15 * sh_width), 0, w - 1)
                        if hips_ok:
                            hy = (int(l_hp.y * h) + int(r_hp.y * h)) // 2
                            torso_h = max(1, hy - sh_y)
                            ty1 = clip(sh_y - int(0.10 * torso_h), 0, h - 1)
                            ty2 = clip(sh_y + int(0.70 * torso_h), 0, h - 1)
                        else:
                            ty1 = clip(sh_y - int(0.05 * h), 0, h - 1)
                            ty2 = clip(sh_y + int(0.25 * h), 0, h - 1)
                        if ty2 > ty1 and tx2 > tx1:
                            torso_roi = (tx1, ty1, tx2, ty2)
                except Exception:
                    torso_roi = None

            have_face = bool(result and getattr(result, 'multi_face_landmarks', None))
            if face_bbox is not None:
                fx1, fy1, fx2, fy2 = face_bbox
                fw = max(1, fx2 - fx1)
                fh = max(1, fy2 - fy1)
                if torso_roi is not None:
                    chest_roi = torso_roi
                else:
                    def clip(v, lo, hi):
                        return max(lo, min(hi, v))
                    cy1 = clip(int(fy2 + 0.15 * fh), 0, h - 1)
                    cy2 = clip(int(min(h - 1, fy2 + 2.0 * fh)), 0, h - 1)
                    cx1 = clip(int(max(0, fx1 - 0.5 * fw)), 0, w - 1)
                    cx2 = clip(int(min(w - 1, fx2 + 0.5 * fw)), 0, w - 1)
                    chest_roi = (cx1, cy1, cx2, cy2)
                expand = max(0.0, args.cig_face_roi_expand)
                ex1 = max(0, int(max(0, fx1 - expand * fw)))
                ey1 = max(0, int(max(0, fy1 - expand * fh)))
                ex2 = min(w - 1, int(min(w - 1, fx2 + expand * fw)))
                ey2 = min(h - 1, int(min(h - 1, fy2 + expand * fh)))
                face_exp_roi = (ex1, ey1, ex2, ey2)
            else:
                if torso_roi is not None:
                    chest_roi = torso_roi
                else:
                    chest_roi = (0, int(h * 0.5), w - 1, h - 1)
                face_exp_roi = (0, 0, w - 1, h - 1)

            # keep track of closest hand box to the mouth (normalized distance)
            min_hand_dist_norm = None
            for xyxy, name, conf in dets:
                nname = name.lower().replace(' ', '_')
                x1, y1, x2, y2 = xyxy
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if getattr(args, 'yolo_labels_overlay', False):
                    yolo_names_this_frame.append((name, conf))
                # Note raw cigarette presence (hand or mouth) even before ROI/face gating
                if ('cigarette' in nname) and (('mouth' in nname) or ('hand' in nname)):
                    raw_cig_present = True
                def in_rect(rc):
                    return (rc[0] <= cx <= rc[2]) and (rc[1] <= cy <= rc[3])
                chest_ok = True if args.no_yolo_gating else (in_rect(chest_roi) or rect_iou(xyxy, chest_roi) > 0.10)
                # Face gate: if required, demand an actual face, not just ROI
                if args.no_yolo_gating:
                    face_ok = True
                else:
                    face_ok = have_face and (in_rect(face_exp_roi) or rect_iou(xyxy, face_exp_roi) > 0.05)

                # Optional box drawing for debugging and validation
                if draw_boxes:
                    try:
                        # Color palette now constrained to { red (0,0,255) and yellow (0,255,255) }.
                        # All previous non-red/non-yellow classes now use red for simplicity.
                        color = (0, 0, 255)
                        if 'no_seatbelt' in nname: color = (0, 0, 255)          # red (critical)
                        elif 'seatbelt' in nname: color = (0, 0, 255)
                        elif 'cigarette_mouth' in nname: color = (0, 0, 255)
                        elif 'cigarette_hand' in nname: color = (0, 0, 255)
                        elif 'no_cigarette' in nname: color = (0, 0, 255)
                        elif 'eye_closed' in nname: color = (0, 0, 255)
                        elif 'eye_open' in nname: color = (0, 255, 255)         # yellow (kept)
                        # Hide cigarette_* and seatbelt* rectangles entirely (user preference)
                        is_cig = ('cigarette_mouth' in nname) or ('cigarette_hand' in nname)
                        is_belt = ('seatbelt' in nname) or ('no_seatbelt' in nname)
                        if (not is_cig) and (not is_belt):
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        # Skip text for seatbelt/cigarette boxes by default to avoid duplication with top-left alerts
                        belt_or_cig = ('cigarette' in nname) or ('seatbelt' in nname) or ('no_seatbelt' in nname)
                        if (not belt_or_cig) or getattr(args, 'box_label_include_belt_smoke', False):
                            cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except Exception:
                        pass
                if nname == 'no_seatbelt' and chest_ok:
                    no_seatbelt_hit = True
                    no_seatbelt_conf_max = max(no_seatbelt_conf_max, conf)
                elif nname == 'seatbelt_worn' and chest_ok:
                    seatbelt_worn_hit = True
                    seatbelt_worn_conf_max = max(seatbelt_worn_conf_max, conf)
                elif nname == 'no_cigarette' and face_ok:
                    nocig_hit = True
                elif nname == 'eye_closed' and face_ok:
                    yolo_eye_closed_hit = True
                elif nname == 'eye_open' and face_ok:
                    yolo_eye_open_hit = True
                if ('cigarette' in nname) and (('mouth' in nname) or ('hand' in nname)) and face_ok:
                    # If face is required for cigarette classes and not present, skip early
                    if getattr(args, 'require_face_for_cig', True) and (not have_face):
                        continue
                    allow_smoke = True
                    if have_mouth:
                        # distance from mouth center to the nearest point on bbox (more permissive than center distance)
                        nx = min(max(mouth_cx, x1), x2)
                        ny = min(max(mouth_cy, y1), y2)
                        dist = math.hypot(nx - mouth_cx, ny - mouth_cy)
                        thr_mouth = float(args.cig_mouth_radius) * face_h_for_mouth
                        thr_hand = float(args.cig_hand_radius) * face_h_for_mouth
                        dn_current = (dist / float(face_h_for_mouth)) if face_h_for_mouth > 0 else 1.0
                        if 'mouth' in nname:
                            allow_smoke = dist <= thr_mouth
                        else:
                            allow_smoke = dist <= thr_hand
                        # store hand min distance (normalized) for later mouth escalation
                        if 'hand' in nname and face_h_for_mouth > 0:
                            dn = dist / float(face_h_for_mouth)
                            min_hand_dist_norm = dn if (min_hand_dist_norm is None) else min(min_hand_dist_norm, dn)
                    # Geometry/brightness constraints to reject finger-only false positives
                    wbox = max(1, x2 - x1)
                    hbox = max(1, y2 - y1)
                    long_side = max(wbox, hbox)
                    short_side = max(1, min(wbox, hbox))
                    aspect = long_side / short_side
                    # approximate face area using face height² (robust to tilt)
                    face_area = float(face_h_for_mouth * face_h_for_mouth) if face_h_for_mouth > 0 else float(h * h * 0.35 * 0.35)
                    area_frac = (wbox * hbox) / max(1.0, face_area)
                    # Suppress near-camera false positives by frame area
                    frame_area = float(max(1, w * h))
                    if ((wbox * hbox) / frame_area) > float(args.cig_max_area_frame_frac):
                        allow_smoke = False
                    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    mean_b = 255.0
                    white_frac = 1.0
                    if roi.size > 0:
                        try:
                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            mean_b = float(gray.mean())
                            # compute white-ish pixel fraction using HSV (low S, high V)
                            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            H, S, V = cv2.split(hsv)
                            # thresholds depend on class (hand/mouth), set below
                            white_frac = 1.0  # default if not computed
                        except Exception:
                            mean_b = 255.0
                            white_frac = 1.0
                    if 'hand' in nname:
                        if aspect < float(args.cig_min_aspect_hand):
                            allow_smoke = False
                        if area_frac > float(args.cig_max_area_frac_hand):
                            allow_smoke = False
                        # brightness threshold (relax when very close to mouth)
                        min_b_req = float(args.cig_min_brightness_hand)
                        try:
                            if have_mouth and (face_h_for_mouth > 0):
                                # use per-detection normalized distance if available
                                dn_local = dn_current if 'dn_current' in locals() else None
                                if (dn_local is not None) and (dn_local <= float(args.cig_relax_dist_norm)):
                                    min_b_req = min(min_b_req, float(args.cig_relax_brightness_hand))
                        except Exception:
                            pass
                        if mean_b < min_b_req:
                            allow_smoke = False
                        # whiteness gating for hand
                        try:
                            if roi.size > 0:
                                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                S = hsv[:, :, 1]
                                V = hsv[:, :, 2]
                                sat_ok = S <= int(args.cig_max_saturation_hand)
                                val_ok = V >= int(min_b_req)
                                mask = np.logical_and(sat_ok, val_ok)
                                white_frac = float(mask.sum()) / float(mask.size)
                                # relax requirement if very close to mouth
                                min_required = float(args.cig_min_white_frac_hand)
                                dn_local2 = dn_current if 'dn_current' in locals() else None
                                if (dn_local2 is not None) and (dn_local2 <= float(args.cig_relax_dist_norm)):
                                    min_required = min(min_required, float(args.cig_relax_white_frac_hand))
                                if white_frac < min_required:
                                    allow_smoke = False
                        except Exception:
                            pass
                    else:  # mouth
                        if aspect < float(args.cig_min_aspect_mouth):
                            allow_smoke = False
                        if area_frac > float(args.cig_max_area_frac_mouth):
                            allow_smoke = False
                        if mean_b < float(args.cig_min_brightness_mouth):
                            allow_smoke = False
                        # whiteness gating for mouth is optional (disabled by default to reduce false negatives)
                        if args.cig_enable_mouth_whiteness:
                            try:
                                if roi.size > 0:
                                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                    S = hsv[:, :, 1]
                                    V = hsv[:, :, 2]
                                    sat_ok = S <= int(args.cig_max_saturation_mouth)
                                    val_ok = V >= int(args.cig_min_brightness_mouth)
                                    mask = np.logical_and(sat_ok, val_ok)
                                    white_frac = float(mask.sum()) / float(mask.size)
                                    if white_frac < float(args.cig_min_white_frac_mouth):
                                        allow_smoke = False
                            except Exception:
                                pass
                        # near-mouth easy accept: if extremely close to mouth, bypass filters
                        try:
                            if have_mouth and (face_h_for_mouth > 0):
                                dn_local = dn_current if 'dn_current' in locals() else None
                                if (dn_local is not None) and (dn_local <= float(args.cig_mouth_easy_norm)):
                                    allow_smoke = True
                        except Exception:
                            pass
                    if allow_smoke:
                        smoking_hit = True
                        if 'mouth' in nname:
                            smoking_mouth_hit = True
                        # cigarette_hand detection removed - only mouth detection now
                    # periodic debug print to match original behavior
                    if args.debug_yolo and (not args.quiet):
                        now_dbg = time.time()
                        if (now_dbg - last_smoking_debug) > 0.5:
                            print(f"YOLO smoking class: {name} {conf:.2f}")
                            last_smoking_debug = now_dbg

            # NOTE: cigarette_hand detection removed - only using YOLO cigarette_Mouth
            # Smoking detection now relies on repetitive hand-to-mouth movement analysis
            
            # PRIORITY: If mobile call detected, suppress all cigarette detections
            if mobile_call_candidate_frame:
                smoking_mouth_hit = False
                smoking_hit = False


            # ========== IMPROVED SEATBELT DETECTION LOGIC ==========
            now_tt = time.time()
            
            # Track raw detection results
            no_seatbelt_conf = no_seatbelt_conf_max if no_seatbelt_hit else 0.0
            seatbelt_worn_conf = seatbelt_worn_conf_max if seatbelt_worn_hit else 0.0
            
            # Determine current frame's detection
            current_detection = None  # 'worn', 'no', or None
            
            if no_seatbelt_hit and seatbelt_worn_hit:
                # Both detected - use confidence margin
                if (no_seatbelt_conf - seatbelt_worn_conf) >= args.seatbelt_conf_margin:
                    current_detection = 'no'
                elif (seatbelt_worn_conf - no_seatbelt_conf) >= args.seatbelt_conf_margin:
                    current_detection = 'worn'
            elif no_seatbelt_hit:
                current_detection = 'no'
            elif seatbelt_worn_hit:
                current_detection = 'worn'
            
            # Initialize state tracking if first run
            if 'seatbelt_state' not in globals():
                globals()['seatbelt_state'] = 'unknown'
                globals()['seatbelt_candidate'] = None
                globals()['seatbelt_confirm_count'] = 0
                globals()['seatbelt_last_seen'] = 0.0
            
            # Update last seen time if detection exists
            if current_detection is not None:
                globals()['seatbelt_last_seen'] = now_tt
            
            # State machine parameters
            CONFIRM_FRAMES_WORN_TO_NO = 3   # Fast for safety
            CONFIRM_FRAMES_NO_TO_WORN = 5   # Slower to avoid false positives
            HOLD_DURATION = float(args.seatbelt_hold_secs)
            
            seatbelt_state = globals()['seatbelt_state']
            seatbelt_candidate = globals()['seatbelt_candidate']
            seatbelt_confirm_count = globals()['seatbelt_confirm_count']
            seatbelt_last_seen = globals()['seatbelt_last_seen']
            
            # Check if in hold period
            in_hold_period = (current_detection is None) and ((now_tt - seatbelt_last_seen) < HOLD_DURATION)
            
            if current_detection is not None:
                # Have detection this frame
                if current_detection == seatbelt_candidate:
                    # Same as candidate - increment
                    seatbelt_confirm_count += 1
                    
                    # Determine confirmation threshold
                    if seatbelt_state == 'worn' and current_detection == 'no':
                        confirm_threshold = CONFIRM_FRAMES_WORN_TO_NO
                    elif seatbelt_state == 'no' and current_detection == 'worn':
                        confirm_threshold = CONFIRM_FRAMES_NO_TO_WORN
                    else:
                        confirm_threshold = CONFIRM_FRAMES_NO_TO_WORN
                    
                    # Check if confirmed
                    if seatbelt_confirm_count >= confirm_threshold:
                        old_state = seatbelt_state
                        seatbelt_state = current_detection
                        seatbelt_candidate = None
                        seatbelt_confirm_count = 0
                        
                        # Trigger alert
                        if seatbelt_state == 'no':
                            if (now_tt - last_alert_times['NO SEATBELT'] >= 0.8):
                                if no_seatbelt_print_count < SEATBELT_MAX_ALERTS:
                                    add_alert("NO SEATBELT")
                                    no_seatbelt_print_count += 1
                                    no_seatbelt_alert_count += 1
                                    optm_print("Seatbelt", f"NO SEATBELT confirmed ({no_seatbelt_alert_count}/{SEATBELT_MAX_ALERTS})")
                                last_alert_times['NO SEATBELT'] = now_tt
                                last_seatbelt_emit = 0.0
                                burst_until_ts = max(burst_until_ts, now_tt + 0.6)
                        
                        elif seatbelt_state == 'worn':
                            if (now_tt - last_alert_times['Seatbelt Worn'] >= 0.8):
                                if seatbelt_worn_print_count < SEATBELT_MAX_ALERTS:
                                    add_alert("Seatbelt Worn")
                                    seatbelt_worn_print_count += 1
                                    dms_print("Seatbelt", f"Seatbelt Worn confirmed ({seatbelt_worn_print_count}/{SEATBELT_MAX_ALERTS})")
                                last_alert_times['Seatbelt Worn'] = now_tt
                                last_seatbelt_emit = 0.0
                                burst_until_ts = max(burst_until_ts, now_tt + 0.6)
                
                elif current_detection != seatbelt_state:
                    # New candidate detected
                    seatbelt_candidate = current_detection
                    seatbelt_confirm_count = 1
            
            elif in_hold_period:
                # No detection but within hold period - keep state
                pass
            
            else:
                # No detection and outside hold period
                if seatbelt_state != 'unknown':
                    seatbelt_state = 'unknown'
                    seatbelt_candidate = None
                    seatbelt_confirm_count = 0
            
            # Update global state
            globals()['seatbelt_state'] = seatbelt_state
            globals()['seatbelt_candidate'] = seatbelt_candidate
            globals()['seatbelt_confirm_count'] = seatbelt_confirm_count
            globals()['seatbelt_last_seen'] = seatbelt_last_seen
            
            # Update legacy variable for compatibility
            seatbelt_status = seatbelt_state
            # ========== END IMPROVED SEATBELT LOGIC ==========

            if 'smoking' not in yolo_counters:
                yolo_counters['smoking'] = 0
            if smoking_hit:
                yolo_counters['smoking'] += 1
            else:
                yolo_counters['smoking'] = 0
            
            # Old seatbelt detection logic removed - using improved state machine above
        # (seatbelt periodic re-emit moved outside the YOLO block below)

            # If face landmarks missing, optionally reflect YOLO eye state
            try:
                if not (result and getattr(result, 'multi_face_landmarks', None)):
                    if yolo_eye_closed_hit and (now_tt - last_alert_times['Eye Closed'] >= YOLO_ALERT_COOLDOWN):
                        add_alert("Eye Closed"); last_alert_times['Eye Closed'] = now_tt
                    elif yolo_eye_open_hit and (now_tt - last_alert_times['Eye Open'] >= YOLO_ALERT_COOLDOWN):
                        add_alert("Eye Open"); last_alert_times['Eye Open'] = now_tt
                # If we lost the face for longer than threshold, clear recognized driver
                # Absence-based clearing (use per-frame presence flag for reliability)
            except Exception:
                pass

            # Apply short holds to reduce flicker
            if smoking_mouth_hit:
                mouth_hold_left = int(args.cig_hold_frames_mouth)
            elif mouth_hold_left > 0:
                smoking_mouth_hit = True
                mouth_hold_left -= 1

            # NEW SMOKING DETECTION: Simple counter-based detection
            # If hand moved to mouth 2+ times, trigger smoking alert
            
            # Check if we have enough counts to confirm smoking behavior
            num_cycles = hand_mouth_counter
            
            # Suppress smoking during phone call
            smoking_blocked = mobile_call_frame
            
            if num_cycles >= SMOKING_CYCLE_THRESHOLD and not smoking_alerted and not smoking_blocked:
                # Repetitive movement detected → likely smoking
                add_alert("Possible Smoking")
                last_alert_times['Possible Smoking'] = now_tt
                smoking_alerted = True
                burst_until_ts = max(burst_until_ts, now_tt + 0.8)
                hand_mouth_counter = 0  # Reset counter immediately after alert to prevent re-trigger
            
            # Reset smoking alert when hand-to-mouth movements stop
            if num_cycles == 0 and not hand_near_mouth_now:
                smoking_alerted = False
                hand_mouth_counter = 0  # Reset counter when no activity
            
            # When any smoking evidence is present this frame, suppress 'No Cigarette' label immediately
            if (smoking_mouth_hit or smoking_hand_hit):
                smoke_recent_until_ts = max(smoke_recent_until_ts, now_tt + 1.2)
                try:
                    if 'No Cigarette' in active_alerts:
                        del active_alerts['No Cigarette']
                except Exception:
                    pass
                nocig_streak = 0; nosignal_streak = 0
            # If only raw detections exist (outside gating), still suppress 'No Cigarette' briefly to avoid contradiction
            elif raw_cig_present:
                smoke_recent_until_ts = max(smoke_recent_until_ts, now_tt + 0.8)
                try:
                    if 'No Cigarette' in active_alerts:
                        del active_alerts['No Cigarette']
                except Exception:
                    pass

            # Old "Smoking detected" alert removed - now using "Possible Smoking" based on hand-to-mouth movement
            # The new detection analyzes repetitive hand-to-mouth cycles (smoker behavior pattern)
            
            # Stabilize smoking status with small streaks and emit periodically for visibility
            smoking_mouth_streak = smoking_mouth_streak + 1 if smoking_mouth_hit else 0
            nocig_streak = nocig_streak + 1 if ((not smoking_hit) and nocig_hit) else 0
            # When face is visible but neither smoking nor explicit no_cig detections appear for a while, gently decay to 'none'
            if (result and getattr(result, 'multi_face_landmarks', None)) and (not smoking_hit) and (not nocig_hit):
                nosignal_streak += 1
            else:
                nosignal_streak = 0
            new_status = smoke_status
            if smoking_mouth_streak >= 1:
                new_status = 'mouth'
            elif nocig_streak >= max(1, int(args.cig_none_min_frames)) or nosignal_streak >= max(0, int(args.cig_decay_frames)):
                new_status = 'none'
            status_changed = (new_status != smoke_status)
            if status_changed:
                smoke_status = new_status
            # Disable periodic re-emit spam - cigarette alert triggers once when confirmed
            # (Similar to seatbelt 'worn' logic - no need for continuous alerts when smoking)
            # The initial "Possible Smoking" alert above is sufficient

    # Periodic re-emit of seatbelt status (DISABLED - trigger once only)
        # Both 'Worn' and 'NO SEATBELT' now trigger ONCE at detection
        # Screen overlay remains visible for ALERT_DURATION (3s) providing sufficient visibility
        # This eliminates console log spam while maintaining clear on-screen warnings
        # Note: Initial alerts still triggered in YOLO detection block (lines ~2330-2365)
        pass

        # NEW Mobile Call Detection Logic:
        # Require hand near BOTH ear AND face for sustained 3 seconds
        # This reduces false positives from brief gestures
        now_phone = time.time()
        
        if mobile_call_candidate_frame and hand_at_ear_and_face:
            # Hand is near ear AND face → potential phone call
            if mobile_call_start_time is None:
                mobile_call_start_time = now_phone  # Start timing
            else:
                # Check if sustained for required duration
                call_duration = now_phone - mobile_call_start_time
                if call_duration >= MOBILE_CALL_DURATION:
                    # Confirmed: sustained for 3+ seconds
                    if not mobile_call_alerted and (now_phone - last_mobile_call_alert_time >= PHONE_ALERT_COOLDOWN):
                        add_alert("Likely Mobile Call")
                        mobile_call_alerted = True
                        last_mobile_call_alert_time = now_phone
                        mobile_call_frame = True  # Flag for this frame
                        smoking_alerted = False
        else:
            # Hand moved away from ear/face → reset
            mobile_call_start_time = None
            mobile_call_alerted = False
            mobile_call_frame = False
        
        # Texting detection (unchanged)
        if texting_candidate_frame:
            texting_confirm_streak += 1
        else:
            texting_confirm_streak = 0
        
        if texting_confirm_streak >= PHONE_CONFIRM_FRAMES:
            if not texting_alerted and (now_phone - last_texting_alert_time >= PHONE_ALERT_COOLDOWN):
                add_alert("Texting Detected")
                texting_alerted = True
                last_texting_alert_time = now_phone
        
        # Reset texting alert when hands move away
        if not texting_candidate_frame and texting_confirm_streak == 0:
            texting_alerted = False


        # Process overlay queue - move alerts from queue to active_alerts after delay
        now = time.time()
        ready_alerts = [item for item in overlay_queue if now >= item['display_after']]
        for item in ready_alerts:
            active_alerts[item['key']] = item['time']
            overlay_queue.remove(item)
        
        # Expire old alerts from overlay
        expired = [k for k, t in active_alerts.items() if now - t > ALERT_DURATION]
        for k in expired:
            del active_alerts[k]

        if args.fps_interval and args.fps_interval > 0 and (not args.quiet):
            frame_count += 1
            if frame_count % args.fps_interval == 0:
                nowp = time.time()
                fps = args.fps_interval / (nowp - last_t)
                # dms_print("Performance", f"FPS ~ {fps:.1f}")  # DISABLED - don't show FPS in logs
                last_t = nowp
                # Auto throttle YOLO if requested
                if getattr(args, 'auto_yolo_skip', False):
                    fps_ema = fps if fps_ema is None else (0.5 * fps_ema + 0.5 * fps)
                    try:
                        if fps_ema < 9.0 and dynamic_skip < 3:
                            dynamic_skip += 1
                        elif fps_ema > 12.0 and dynamic_skip > 1:
                            dynamic_skip -= 1
                    except Exception:
                        pass

        # Display authenticated driver name at top-left (if authenticated)
        if authenticated_driver:
            driver_text = f"Driver: {authenticated_driver} (ID: {driver_id})"
            cv2.putText(frame, driver_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            alert_offset = 58  # Start alerts below driver name
        else:
            alert_offset = 30  # Start alerts at top if no authentication
        
        # Draw active alerts - red/yellow colors based on severity
        for i, msg in enumerate(list(active_alerts.keys())[:12]):
            # Red for critical/severe, yellow for moderate/warning
            color = (0, 0, 255)  # red (default for severe/critical)
            if "Moderate" in msg or "Alert" in msg or "Warning" in msg:
                color = (0, 255, 255)  # yellow for moderate
            disp = msg
            cv2.putText(frame, disp, (10, alert_offset + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display eye openness percentage (instantaneous) - like DMSv8
        if not calibration_mode and result and getattr(result, 'multi_face_landmarks', None):
            if 'current_eye_openness' in locals():
                # Always green color like DMSv8
                pct_color = (0, 255, 0)
                cv2.putText(frame, f"Eye Open: {current_eye_openness:.1f}%", (w - 230, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, pct_color, 2)

        # Draw raw YOLO labels (optional) to verify the 7 classes visibly
        try:
            show_yolo_labels = bool(dets) and getattr(args, 'yolo_labels_overlay', False)
            if show_yolo_labels:
                # Deduplicate by name keeping highest conf
                best_by_name = {}
                for nm, cf in yolo_names_this_frame:
                    if (nm not in best_by_name) or (cf > best_by_name[nm]):
                        best_by_name[nm] = cf
                base_y = 30 + min(12, len(active_alerts)) * 28 + 8
                idx = 0
                for nm, cf in sorted(best_by_name.items(), key=lambda kv: -kv[1]):
                    # Use class-aware colors for readability; cigarette classes in orange
                    nm_lower = nm.lower()
                    if 'cigarette' in nm_lower and 'no_cigarette' not in nm_lower:
                        col = (0, 0, 255)
                    elif 'no_seatbelt' in nm_lower:
                        col = (0, 0, 255)
                    elif 'seatbelt' in nm_lower:
                        col = (0, 0, 255)
                    elif 'eye_closed' in nm_lower:
                        col = (0, 0, 255)
                    elif 'eye_open' in nm_lower:
                        col = (0, 255, 255)     # yellow retained
                    else:
                        col = (0, 0, 255)
                    # HUD filter rules:
                    # - Show cigarette_Hand/Mouth by default (orange), but hide 'no_Cigarette' by default.
                    # - Hide seatbelt entries unless --yolo_overlay_include_belt_smoke is passed.
                    is_belt = ('seatbelt' in nm_lower) or ('no_seatbelt' in nm_lower)
                    is_no_cig = ('no_cigarette' in nm_lower)
                    if is_belt and not getattr(args, 'yolo_overlay_include_belt_smoke', False):
                        continue
                    if is_no_cig and not getattr(args, 'yolo_overlay_include_belt_smoke', False):
                        continue
                    # Optional confidence on demand; default is no confidence.
                    label_txt = nm if not getattr(args, 'yolo_overlay_conf', False) else f"{nm} {cf:.2f}"
                    cv2.putText(frame, label_txt, (10, base_y + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    idx += 1
                # Removed skip overlay per user request.
        except Exception:
            pass

        # Optional retina-style HUD: show State/EAR/Blinks without altering existing alerts
        try:
            if getattr(args, 'retina_overlay', False):
                # Derive a simple OPEN/CLOSED state from current eye signal
                final_state = 'OPEN' if (prev_eye_state == 'open') else 'CLOSED'
                status_color = (0, 0, 255)  # always red for OPEN/CLOSED states now
                # Estimate EAR using last computed left/right EAR if available
                # Fallback to 0.00 if not in scope
                try:
                    ear_txt = f"{avg_ear:.2f}"
                except Exception:
                    ear_txt = "0.00"
                # Draw in the top-right area to avoid overlay collision
                x0 = max(10, frame.shape[1] - 260)
                cv2.putText(frame, f"State: {final_state}", (x0, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"EAR: {ear_txt}", (x0, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"Blinks: {blink_total}", (x0, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        except Exception:
            pass
        
        # KSS HUD Display (always visible for AIS 184 monitoring)
        try:
            if 'kss_score' in locals() and 'kss_confidence' in locals():
                # Get color based on KSS level (green <7, yellow 7-8, red >=8)
                kss_color = kss_alert_manager.get_alert_color(kss_score)
                kss_label = kss_calculator.get_kss_label(kss_score)
                
                # Display KSS score in bottom-right corner
                # KSS HUD display removed per user request
        except Exception as e:
            pass

        # Session management and risk accumulation (end-of-loop)
        try:
            if risk_enabled:
                # Periodic accumulation of composite risk (time-weighted)
                now_r = time.time()
                if (now_r - last_risk_update_ts) >= float(getattr(args, 'risk_update_interval', 1.0) or 1.0):
                    lvl = 0
                    for _, riskv, _ in risk_events:
                        lvl = max(lvl, int(riskv or 0))
                    current_risk_level = lvl
                    dt = now_r - last_risk_update_ts
                    last_risk_update_ts = now_r
        except Exception:
            pass

        # Minimal on-screen YOLO status to diagnose missing detections
        try:
            # Bottom-left YOLO status is off by default; enable with --yolo_status_overlay
            if getattr(args, 'yolo_status_overlay', False):
                now_ts = time.time()
                yolo_ok = (yolo_worker is not None) and (last_yolo_result_time > 0) and ((now_ts - last_yolo_result_time) < 2.0)
                status_txt = "active" if yolo_ok else ("no results" if (yolo_worker is not None) else "not loaded")
                # Active = red; intermediate (worker but no results) = yellow; not loaded = red
                color = (0, 0, 255) if yolo_ok else ((0, 255, 255) if yolo_worker is not None else (0, 0, 255))
                cv2.putText(frame, status_txt, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception:
            pass 

        # Risk HUD: simple color bar (green/yellow/red) when risk enabled
        try:
            if risk_enabled:
                # Draw a small bar at top-left showing current_risk_level (0..3)
                bar_x, bar_y, bar_w, bar_h = 10, 10, 120, 12
                # background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (40,40,40), -1)
                # choose color
                col = (0,200,0) if current_risk_level <= 1 else ((0,255,255) if current_risk_level == 2 else (0,0,255))
                fill_w = int(bar_w * (current_risk_level / 3.0))
                if fill_w > 0:
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), col, -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (200,200,200), 1)
                cv2.putText(frame, f"Risk {current_risk_level}", (bar_x+126, bar_y+bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                # Optional medical badge below the bar when chest clutching is active
                try:
                    if medical_enabled and chest_clutch_active:
                        bx, by = bar_x, bar_y + bar_h + 6
                        bw, bh = 44, 16
                        # red badge background with border
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,0,255), -1)
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (240,240,240), 1)
                        cv2.putText(frame, "MED", (bx+6, by+bh-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                except Exception:
                    pass
        except Exception:
            pass

        # Send frame to remote server or display locally
        if frame_sender is not None:
            # Send to remote TCP server
            frame_sender.send_frame(frame)
        elif no_stream_mode:
            # No stream mode - log data only (no display, no streaming)
            # Alerts and data already printed to terminal via dms_print()
            pass  # No frame display or sending
        else:
            # Local display mode
            cv2.imshow("DMS Integrated (refactored)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        # All enrollment-related key handlers removed (runtime enrollment disabled)
        
        # End frame timing
        if _LAT:
            lat.end_frame()

        fid += 1

    cap.release()
    if yolo_worker:
        yolo_worker.stop()
    if frame_sender is not None:
        try:
            frame_sender.close()
        except Exception:
            pass
    # No DB teardown in JSON-only build
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
