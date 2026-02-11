"""
Main Driver Monitoring System (DMS) Controller - IMX Board NPU Version
Uses vela-optimized models on NPU for face detection and recognition
Uses standard INT8 model on CPU for landmark detection

Models:
- scrfd_500m_full_int8_vela.tflite (NPU) - Face detection
- fr_int8_velaS.tflite (NPU) - Face recognition  
- face_landmark_192_int8.tflite (CPU) - FaceMesh landmarks
"""

import cv2
import numpy as np
import time
import threading
import queue
import subprocess
import sys
import os
import re
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# Fix for embedded systems without display
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# YOLO runtime imports
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

from scrfd_detector_board import SCRFDDetector, align_face
from recognize_v2_board import FaceRecognizer


# ============================================================
# MJPEG Streaming Server for VLC viewing
# ============================================================

import http.server
import socketserver
from io import BytesIO

class MJPEGStreamHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming to VLC"""
    
    def do_GET(self):
        if self.path == '/video':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            try:
                while True:
                    if hasattr(self.server, 'current_frame') and self.server.current_frame is not None:
                        # Encode frame to JPEG
                        ret, jpeg = cv2.imencode('.jpg', self.server.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        if ret:
                            self.wfile.write(b"--jpgboundary\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logs


class MJPEGServer:
    """MJPEG streaming server for network viewing"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.current_frame = None
    
    def start(self):
        """Start MJPEG server in background thread"""
        try:
            self.server = socketserver.ThreadingTCPServer((self.host, self.port), MJPEGStreamHandler)
            self.server.current_frame = None
            
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            
            print(f"[MJPEG] Streaming at http://{self.host}:{self.port}/video")
            print(f"[MJPEG] Open in VLC: vlc http://<board-ip>:{self.port}/video")
            return True
        except Exception as e:
            print(f"[MJPEG] Failed to start server: {e}")
            return False
    
    def update_frame(self, frame):
        """Update the frame to be streamed"""
        if self.server is not None:
            self.server.current_frame = frame
    
    def stop(self):
        """Stop MJPEG server"""
        if self.server is not None:
            self.server.shutdown()
            print("[MJPEG] Server stopped")


# ============================================================
# YOLO Utilities and Worker
# ============================================================

DEFAULT_NAMES = [
    "seatbelt_worn",
    "no_seatbelt",
    "cigarette_Hand",
    "cigarette_Mouth",
    "no_Cigarette",
    "unknown_class",  # 6th class for model compatibility
]

def parse_thresholds(th_str: str) -> dict:
    out: dict = {}
    if not th_str:
        return out
    for kv in th_str.split(','):
        kv = kv.strip()
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return out

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def _convert_results_to_dets(res, names: list, class_thresholds: dict, base_conf: float):
    dets = []
    try:
        r0 = res[0] if isinstance(res, list) else res
    except Exception:
        r0 = None
    if r0 is None or getattr(r0, 'boxes', None) is None or len(r0.boxes) == 0:
        return dets
    for b in r0.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        cls_name = names[cls_id] if cls_id < len(names) else f"class{cls_id}"
        norm = _norm_name(cls_name)
        thr = class_thresholds.get(norm, class_thresholds.get(cls_name, base_conf))
        if conf >= thr:
            xyxy = b.xyxy[0].cpu().numpy()
            dets.append({'box': xyxy, 'conf': conf, 'cls': cls_id, 'name': cls_name})
    return dets


class YOLOWorker(threading.Thread):
    def __init__(self, model_path: str, imgsz: int, conf: float, iou: float,
                 class_thresholds: dict, names: list, nms_topk: int = 300, sleep_interval: float = 10.0):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.class_thresholds = class_thresholds
        self.names = names
        self.nms_topk = nms_topk
        self.sleep_interval = sleep_interval
        
        self.use_onnx = model_path.lower().endswith('.onnx')
        self.model = None
        self.ort_session = None
        self.input_name = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_dets = []
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.wake_event = threading.Event()
        
        if not self.validate_model():
            print(f"[YOLO] Warning: model validation failed for {model_path}")
    
    def validate_model(self) -> bool:
        if not os.path.isfile(self.model_path):
            print(f"[YOLO] Model file not found: {self.model_path}")
            return False
        
        if self.use_onnx:
            if not _HAS_ORT:
                print("[YOLO] onnxruntime not available")
                return False
            try:
                self.ort_session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.ort_session.get_inputs()[0].name
                input_shape = self.ort_session.get_inputs()[0].shape
                output_shape = self.ort_session.get_outputs()[0].shape
                
                print(f"[YOLO] ONNX model loaded: {self.model_path}")
                print(f"[YOLO] Input: {self.input_name}, shape: {input_shape}")
                print(f"[YOLO] Output shape: {output_shape}")
                print(f"[YOLO] Outputs: {len(self.ort_session.get_outputs())} tensors")
                
                # Verify class count
                if output_shape and len(output_shape) >= 2:
                    num_values = output_shape[1] if output_shape[1] < output_shape[2] else output_shape[2]
                    expected_classes = num_values - 5
                    print(f"[YOLO] Model expects {expected_classes} classes, you defined {len(self.names)}")
                    
                    if expected_classes != len(self.names):
                        print(f"[YOLO] WARNING: Class count mismatch!")
                
                return True
            except Exception as e:
                print(f"[YOLO] ONNX load error: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            if not _HAS_YOLO:
                print("[YOLO] Ultralytics not available")
                return False
            try:
                self.model = YOLO(self.model_path)
                print(f"[YOLO] Ultralytics model loaded: {self.model_path}")
                return True
            except Exception as e:
                print(f"[YOLO] Ultralytics load error: {e}")
                return False
    
    def submit(self, frame):
        """Store latest frame without blocking or queuing"""
        if not self.running:
            return
        with self.frame_lock:
            self.latest_frame = frame  # Just store reference, no copy needed
    
    def get_latest(self):
        with self.lock:
            return list(self.latest_dets)
    
    def stop(self):
        self.running = False
    
    def run(self):
        self.running = True
        print(f"[YOLO] Worker started - periodic mode ({self.sleep_interval}s intervals)")
        print(f"[YOLO] Using {'ONNX' if self.use_onnx else 'Ultralytics'} backend")
        print(f"[YOLO] Model: {self.model_path}, imgsz={self.imgsz}, conf={self.conf}")
        
        detection_cycle = 0
        
        while self.running:
            # Sleep first, then process
            if detection_cycle > 0:  # Skip sleep on first run
                if not self.wake_event.wait(timeout=self.sleep_interval):
                    pass  # Timeout reached, proceed with detection
                self.wake_event.clear()
            
            if not self.running:
                break
            
            # Get latest frame
            with self.frame_lock:
                frame = self.latest_frame
            
            if frame is None:
                time.sleep(0.5)
                continue
            
            # Run detection
            detection_cycle += 1
            print(f"\n[YOLO] Cycle #{detection_cycle}: Running inference...")
            
            try:
                if self.use_onnx:
                    dets = self._infer_onnx(frame)
                else:
                    res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                           iou=self.iou, verbose=False, device='cpu')
                    dets = _convert_results_to_dets(res, self.names, self.class_thresholds, self.conf)
                
                with self.lock:
                    self.latest_dets = dets
                
                # Compact summary
                if dets:
                    det_summary = {}
                    for d in dets:
                        det_summary[d['name']] = det_summary.get(d['name'], 0) + 1
                    print(f"[YOLO] Found: {det_summary}")
                else:
                    print(f"[YOLO] No detections")
                    
            except Exception as e:
                print(f"[YOLO] Error: {e}")
        
        print("[YOLO] Worker stopped")
    
    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw //= 2
        dh //= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = dh, dh
        left, right = dw, dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, dw, dh
    
    def _infer_onnx(self, frame):
        if self.ort_session is None:
            return []
        
        img0 = frame
        h, w = img0.shape[:2]
        img_size = int(self.imgsz)
        
        # Letterbox preprocessing (reference style)
        img, r, dw, dh = self._letterbox(img0, img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img_rgb.transpose(2, 0, 1)[None].astype('float32') / 255.0
        
        try:
            outputs = self.ort_session.run(None, {self.input_name: x})
            print(f"[YOLO-ONNX] Raw output shape: {outputs[0].shape}")
        except Exception as e:
            print(f"[YOLO] ONNX inference error: {e}")
            return []
        
        dets = []
        nc = len(self.names)
        
        # Reference implementation: robust handling for different output formats
        z = outputs[0]
        
        # Handle 3D output
        if z.ndim == 3 and z.shape[0] == 1:
            # Check if needs transpose: (1, nc+5, num_boxes) -> (1, num_boxes, nc+5)
            # For YOLOv5/v8: output is (1, nc+5, 8400) with objectness
            if z.shape[2] > z.shape[1]:
                z = np.transpose(z, (0, 2, 1))
            z = z[0]  # Remove batch dimension: (num_boxes, nc+5)
        
        print(f"[YOLO-ONNX] Predictions after reshape: {z.shape}")
        
        # Handle (num_boxes, nc+5) format: [cx, cy, w, h, objectness, class0, ..., classN]
        if z.ndim == 2 and z.shape[1] == (nc + 5):
            print(f"[YOLO-ONNX] Detected YOLOv5/v8 format with objectness score")
            boxes = z[:, :4]          # [cx, cy, w, h]
            objectness = z[:, 4:5]    # objectness score
            class_scores = z[:, 5:]   # class probabilities
            
            print(f"[YOLO-ONNX] RAW Objectness - Max: {objectness.max():.4f}, Min: {objectness.min():.4f}")
            print(f"[YOLO-ONNX] RAW Class scores - Max: {class_scores.max():.4f}, Min: {class_scores.min():.4f}")
            
            # Apply sigmoid activation (YOLOv5/v8 outputs are logits)
            objectness = 1.0 / (1.0 + np.exp(-objectness))
            class_scores = 1.0 / (1.0 + np.exp(-class_scores))
            
            print(f"[YOLO-ONNX] SIGMOID Objectness - Max: {objectness.max():.4f}, Min: {objectness.min():.4f}")
            print(f"[YOLO-ONNX] SIGMOID Class scores - Max: {class_scores.max():.4f}, Min: {class_scores.min():.4f}")
            
            # Convert cx,cy,w,h to x1,y1,x2,y2
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            
            # Final confidence = objectness * class_score
            class_ids = np.argmax(class_scores, axis=1)
            max_class_scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
            confs = objectness.flatten() * max_class_scores
            
            print(f"[YOLO-ONNX] Max confidence: {confs.max():.4f}")
            
            # Show top 5
            top_indices = np.argsort(confs)[-5:][::-1]
            print(f"[YOLO-ONNX] Top 5 predictions:")
            for idx in top_indices:
                cid = int(class_ids[idx])
                name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                print(f"  {name}: {confs[idx]:.4f} (obj={objectness[idx,0]:.4f}, cls={max_class_scores[idx]:.4f})")
            
            # Filter by threshold
            for i in range(z.shape[0]):
                conf = float(confs[i])
                cid = int(class_ids[i])
                name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                norm = _norm_name(name)
                thr = self.class_thresholds.get(norm, self.class_thresholds.get(name, self.conf))
                
                if conf < thr:
                    continue
                
                dets.append([x1[i], y1[i], x2[i], y2[i], conf, cid])
        
        # Handle (num_boxes, nc+4) format: [cx, cy, w, h, class0, ..., classN] (no objectness)
        elif z.ndim == 2 and z.shape[1] == (nc + 4):
            print(f"[YOLO-ONNX] Detected format without objectness (nc+4)")
            boxes = z[:, :4]
            scores = z[:, 4:]
            
            print(f"[YOLO-ONNX] Score stats - Max: {scores.max():.4f}, Min: {scores.min():.4f}, Mean: {scores.mean():.4f}")
            
            # Convert cx,cy,w,h to x1,y1,x2,y2
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            
            class_ids = np.argmax(scores, axis=1)
            confs = scores[np.arange(scores.shape[0]), class_ids]
            
            print(f"[YOLO-ONNX] Max confidence: {confs.max():.4f}")
            
            # Show top 5
            top_indices = np.argsort(confs)[-5:][::-1]
            print(f"[YOLO-ONNX] Top 5 predictions:")
            for idx in top_indices:
                cid = int(class_ids[idx])
                name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                print(f"  {name}: {confs[idx]:.4f}")
            
            # Filter by threshold
            for i in range(z.shape[0]):
                conf = float(confs[i])
                cid = int(class_ids[i])
                name = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                norm = _norm_name(name)
                thr = self.class_thresholds.get(norm, self.class_thresholds.get(name, self.conf))
                
                if conf < thr:
                    continue
                
                dets.append([x1[i], y1[i], x2[i], y2[i], conf, cid])
        
        else:
            print(f"[YOLO-ONNX] Unexpected output shape: {z.shape}, expected (N, {nc+5}) or (N, {nc+4})")
            return []
        
        if not dets:
            print(f"[YOLO-ONNX] No detections after threshold filter")
            return []
        
        dets = np.array(dets, dtype=np.float32)
        print(f"[YOLO-ONNX] Detections before NMS: {len(dets)}")
        
        # NMS
        keep = self._nms(dets[:, :4], dets[:, 4], self.iou, top_k=self.nms_topk)
        dets = dets[keep]
        
        print(f"[YOLO-ONNX] After NMS: {len(dets)} detections")
        
        # Convert to original coordinates and output format
        out = []
        for x1_, y1_, x2_, y2_, conf_, cid_ in dets:
            x1o = max(0, int(round((x1_ - dw) / r)))
            y1o = max(0, int(round((y1_ - dh) / r)))
            x2o = max(0, int(round((x2_ - dw) / r)))
            y2o = max(0, int(round((y2_ - dh) / r)))
            
            x1o = max(0, min(w, x1o))
            y1o = max(0, min(h, y1o))
            x2o = max(0, min(w, x2o))
            y2o = max(0, min(h, y2o))
            
            name = self.names[int(cid_)] if 0 <= int(cid_) < len(self.names) else str(int(cid_))
            
            out.append({
                'box': np.array([x1o, y1o, x2o, y2o]),
                'conf': float(conf_),
                'cls': int(cid_),
                'name': name
            })
        
        return out
    
    def _nms(self, boxes, scores, iou_thres=0.45, top_k=300):
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        if top_k > 0:
            order = order[:top_k]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        
        return keep


class DriverMonitoringWrapper:
    """Wrapper for driver monitoring using FaceMesh (CPU)"""
    
    def __init__(self):
        print("[Monitor-Board] Initializing FaceMesh on CPU...")
        
        # Load FaceMesh model (CPU - not vela optimized)
        self.mesh_interpreter = tflite.Interpreter(model_path="face_landmark_192_int8.tflite")
        self.mesh_interpreter.allocate_tensors()
        
        self.mesh_input_details = self.mesh_interpreter.get_input_details()[0]
        self.mesh_output_details = self.mesh_interpreter.get_output_details()
        
        self.mesh_in_scale, self.mesh_in_zero = self.mesh_input_details['quantization']
        self.mesh_out_scale, self.mesh_out_zero = self.mesh_output_details[0]['quantization']
        
        print(f"[Monitor-Board] FaceMesh on CPU - Input: scale={self.mesh_in_scale}, zero={self.mesh_in_zero}")
        
        # Load Eye Detection Model (Vela-optimized for NPU)
        print("[Monitor-Board] Loading eye detection model (NPU)...")
        self.eye_interpreter = tflite.Interpreter(
            model_path="eye_detection_int8_vela.tflite",
            experimental_delegates=[tflite.load_delegate('/usr/lib/libethosu_delegate.so')]
        )
        self.eye_interpreter.allocate_tensors()
        
        self.eye_input_details = self.eye_interpreter.get_input_details()[0]
        self.eye_output_details = self.eye_interpreter.get_output_details()[0]
        
        self.eye_in_scale, self.eye_in_zero = self.eye_input_details['quantization']
        self.eye_out_scale, self.eye_out_zero = self.eye_output_details['quantization']
        
        print(f"[Monitor-Board] Eye Detection (NPU): {self.eye_input_details['shape']}, scale={self.eye_in_scale:.6f}")
        
        # Load Phone Detection Model (Vela-optimized for NPU)
        print("[Monitor-Board] Loading phone detection model (NPU)...")
        self.phone_interpreter = tflite.Interpreter(
            model_path="detect_vela.tflite",
            experimental_delegates=[tflite.load_delegate('/usr/lib/libethosu_delegate.so')]
        )
        self.phone_interpreter.allocate_tensors()
        
        self.phone_input_details = self.phone_interpreter.get_input_details()[0]
        self.phone_output_details = self.phone_interpreter.get_output_details()
        
        print(f"[Monitor-Board] Phone Detection (NPU): {self.phone_input_details['shape']}")
        
        # Phone detection settings
        self.PHONE_CLASS_IDS = [77, 76, 73, 74]  # phone, keyboard, laptop, mouse
        self.phone_confidence_threshold = 0.35
        self.texting_counter = 0
        self.call_counter = 0
        self.phone_threshold = 3
        
        # Landmark indices
        self.LEFT_EYE = [33, 133, 157, 158, 159, 160]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 384]
        self.NOSE_TIP = 1
        
        # Thresholds
        self.ear_threshold = 0.6
        self.mar_threshold = 0.08
        self.eye_closed_frames = 6
        self.yawn_frames = 3
        self.head_turn_threshold = 0.08
        self.ear_baseline = None
        self.ear_history = []
        self.ear_drop_threshold = 0.25
        
        # Counters
        self.eye_closure_counter = 0
        self.blink_counter = 0
        self.yawn_counter = 0
        self.drowsiness_counter = 0
        self.blink_timer = time.time()
        
        # Head pose baseline
        self.baseline_samples = deque(maxlen=30)
        self.head_baseline = None
        self.baseline_ready = False
        
        print("[Monitor-Board] Driver monitoring initialized")
    
    def expand_bbox(self, x1, y1, x2, y2, img_w, img_h, expand_ratio=0.3):
        """Expand bounding box"""
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, int(x1 - w * expand_ratio))
        y1 = max(0, int(y1 - h * expand_ratio))
        x2 = min(img_w, int(x2 + w * expand_ratio))
        y2 = min(img_h, int(y2 + h * expand_ratio))
        return x1, y1, x2, y2
    
    def preprocess_facemesh(self, face_crop):
        """Preprocess face for FaceMesh"""
        face_resized = cv2.resize(face_crop, (192, 192))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_float = face_rgb.astype(np.float32) / 255.0
        face_quantized = (face_float / self.mesh_in_scale) + self.mesh_in_zero
        face_quantized = np.clip(face_quantized, -128, 127).astype(np.int8)
        return np.expand_dims(face_quantized, axis=0)
    
    def get_landmarks(self, face_crop):
        """Extract facial landmarks"""
        try:
            face_input = self.preprocess_facemesh(face_crop)
            self.mesh_interpreter.set_tensor(self.mesh_input_details['index'], face_input)
            self.mesh_interpreter.invoke()
            
            landmarks_raw = self.mesh_interpreter.get_tensor(self.mesh_output_details[0]['index'])
            landmarks = (landmarks_raw.astype(np.float32) - self.mesh_out_zero) * self.mesh_out_scale
            landmarks = landmarks.reshape((468, 3))
            
            landmarks[:, 0] /= 192.0
            landmarks[:, 1] /= 192.0
            
            return landmarks
        except:
            return None
    
    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio"""
        if landmarks is None or len(eye_indices) < 6:
            return 0.0
        
        eye_points = np.array([[landmarks[i, 0], landmarks[i, 1]] for i in eye_indices])
        
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        return (A + B) / (2.0 * C) if C > 0 else 0.0
    
    def detect_eye_state_ml(self, face_crop, landmarks):
        """Detect eye open/closed using MobileNetV2 Vela model"""
        # Skip every other frame to reduce NPU load
        if not hasattr(self, '_eye_frame_count'):
            self._eye_frame_count = 0
        
        self._eye_frame_count += 1
        if self._eye_frame_count % 2 == 0:
            return None, 0.0
        
        try:
            # Resize to 256x256 first, then center crop to 224
            face_256 = cv2.resize(face_crop, (256, 256))
            margin = (256 - 224) // 2
            eye_resized = face_256[margin:margin+224, margin:margin+224]
            
            eye_rgb = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB)
            eye_float = eye_rgb.astype(np.float32) / 255.0  # [0,1] range
            
            # Quantize to INT8
            eye_quantized = (eye_float / self.eye_in_scale) + self.eye_in_zero
            eye_quantized = np.clip(eye_quantized, -128, 127).astype(np.int8)
            eye_quantized = np.expand_dims(eye_quantized, axis=0)
            
            # Run inference
            self.eye_interpreter.set_tensor(self.eye_input_details['index'], eye_quantized)
            self.eye_interpreter.invoke()
            output_int8 = self.eye_interpreter.get_tensor(self.eye_output_details['index'])[0]
            
            # Dequantize output
            output_float = (output_int8.astype(np.float32) - self.eye_out_zero) * self.eye_out_scale
            
            # Softmax
            exp_vals = np.exp(output_float - np.max(output_float))
            softmax = exp_vals / np.sum(exp_vals)
            
            prediction = np.argmax(softmax)
            confidence = softmax[prediction]
            
            is_open = (prediction == 1)  # 0=closed, 1=open
            
            return is_open, confidence
            
        except Exception as e:
            print(f"[EYE-ML-ERROR] {e}")
            return None, 0.0
    
    def detect_phone(self, frame, face_box=None):
        """Detect phone and classify call/texting relative to face"""
        # Skip 2 out of 3 frames to reduce NPU load
        if not hasattr(self, '_phone_frame_count'):
            self._phone_frame_count = 0
        
        self._phone_frame_count += 1
        if self._phone_frame_count % 3 != 0:
            return {
                'possible_texting': self.texting_counter >= self.phone_threshold,
                'likely_mobile_call': self.call_counter >= self.phone_threshold,
                'phone_detected': False,
                'phone_position': None,
                'phone_confidence': 0.0
            }
        
        try:
            h, w = frame.shape[:2]
            
            # Preprocess: resize to 300x300
            resized = cv2.resize(frame, (300, 300))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
            
            # Run inference
            self.phone_interpreter.set_tensor(self.phone_input_details['index'], input_data)
            self.phone_interpreter.invoke()
            
            # Get outputs
            boxes = self.phone_interpreter.get_tensor(self.phone_output_details[0]['index'])[0]
            classes = self.phone_interpreter.get_tensor(self.phone_output_details[1]['index'])[0]
            scores = self.phone_interpreter.get_tensor(self.phone_output_details[2]['index'])[0]
            num_detections = int(self.phone_interpreter.get_tensor(self.phone_output_details[3]['index'])[0])
            
            # Filter for phone detections
            phone_detected = False
            phone_position = None
            phone_confidence = 0.0
            
            for i in range(num_detections):
                class_id = int(classes[i])
                confidence = scores[i]
                
                if class_id in self.PHONE_CLASS_IDS and confidence >= self.phone_confidence_threshold:
                    phone_detected = True
                    phone_confidence = confidence
                    
                    # Determine position [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = boxes[i]
                    phone_center_y = (ymin + ymax) / 2.0
                    
                    # Use face-relative positioning if face detected
                    if face_box is not None:
                        face_y_min = face_box[1] / h
                        face_y_max = face_box[3] / h
                        
                        if phone_center_y <= face_y_max + 0.05:
                            phone_position = 'call'
                        elif phone_center_y > face_y_max + 0.10:
                            phone_position = 'texting'
                        else:
                            phone_position = None
                    
                    break
            
            # Temporal filtering
            if phone_detected and phone_position == 'texting':
                self.texting_counter += 1
                self.call_counter = 0
            elif phone_detected and phone_position == 'call':
                self.call_counter += 1
                self.texting_counter = 0
            else:
                self.texting_counter = max(0, self.texting_counter - 1)
                self.call_counter = max(0, self.call_counter - 1)
            
            return {
                'possible_texting': self.texting_counter >= self.phone_threshold,
                'likely_mobile_call': self.call_counter >= self.phone_threshold,
                'phone_detected': phone_detected,
                'phone_position': phone_position,
                'phone_confidence': phone_confidence
            }
        except Exception as e:
            print(f"[PHONE-ERROR] {e}")
            return {
                'possible_texting': False,
                'likely_mobile_call': False,
                'phone_detected': False,
                'phone_position': None,
                'phone_confidence': 0.0
            }
    
    def calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio"""
        if landmarks is None:
            return 0.0
        
        v_pairs = [(13, 14), (82, 87), (312, 317)]
        verticals = []
        for top, bot in v_pairs:
            if top < len(landmarks) and bot < len(landmarks):
                v_dist = np.linalg.norm(landmarks[top, :2] - landmarks[bot, :2])
                verticals.append(v_dist)
        
        h_left, h_right = 61, 291
        if h_left < len(landmarks) and h_right < len(landmarks):
            horizontal = np.linalg.norm(landmarks[h_left, :2] - landmarks[h_right, :2])
            if horizontal > 0 and verticals:
                return sum(verticals) / (len(verticals) * horizontal)
        
        return 0.0
    
    def analyze(self, frame, face_box):
        """Analyze driver behavior"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_box
        
        crop_x1, crop_y1, crop_x2, crop_y2 = self.expand_bbox(x1, y1, x2, y2, w, h)
        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if face_crop.size == 0:
            return None
        
        landmarks = self.get_landmarks(face_crop)
        if landmarks is None:
            return None
        
        # Calculate metrics
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self.calculate_mar(landmarks)
        
        # Track EAR history for adaptive detection
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)
        
        # Adaptive EAR baseline
        if len(self.ear_history) >= 10:
            self.ear_baseline = np.mean(self.ear_history[-30:])
        
        ear_threshold = self.ear_baseline if self.ear_baseline else self.ear_threshold
        
        # Head pose
        head_x = landmarks[self.NOSE_TIP, 0]
        head_y = landmarks[self.NOSE_TIP, 1]
        
        # Adaptive baseline
        if not self.baseline_ready:
            self.baseline_samples.append((head_x, head_y))
            if len(self.baseline_samples) >= 30:
                head_positions = np.array(list(self.baseline_samples))
                self.head_baseline = np.median(head_positions, axis=0)
                self.baseline_ready = True
                print(f"[Monitor-Board] Baseline calibrated")
        
        # Detect alerts
        alerts = []
        eye_closed_level = 0
        head_droop_level = 0
        head_turn_level = 0
        head_tilt_level = 0
        yawn_detected = False
        
        # ML-based eye detection (Vela-optimized NPU)
        is_open, confidence = self.detect_eye_state_ml(face_crop, landmarks)
        
        if is_open is not None and confidence >= 0.60:
            eyes_closed = not is_open
        else:
            # Fallback: use EAR if ML fails
            if self.ear_baseline and self.ear_baseline > 0:
                ear_drop_percent = (self.ear_baseline - avg_ear) / self.ear_baseline
                eyes_closed = ear_drop_percent > self.ear_drop_threshold
            else:
                eyes_closed = avg_ear < ear_threshold
        
        if eyes_closed:
            self.eye_closure_counter += 1
            
            if self.eye_closure_counter > 30:
                alerts.append(("Severe", "Eyes Closed Too Long"))
                eye_closed_level = 2
            elif self.eye_closure_counter > self.eye_closed_frames:
                alerts.append(("Warning", "Eyes Closed"))
                eye_closed_level = 1
        else:
            if 2 <= self.eye_closure_counter < self.eye_closed_frames:
                self.blink_counter += 1
            self.eye_closure_counter = 0
        
        # Blink rate
        if time.time() - self.blink_timer > 60:
            if self.blink_counter >= 30:
                alerts.append(("Warning", "High Blink Rate"))
            self.blink_counter = 0
            self.blink_timer = time.time()
        
        # Yawning
        if mar > self.mar_threshold:
            self.yawn_counter += 1
            if self.yawn_counter > self.yawn_frames:
                alerts.append(("Warning", "Yawning Detected"))
                yawn_detected = True
        else:
            self.yawn_counter = 0
        
        # Head pose analysis
        if self.baseline_ready:
            head_x_offset = abs(head_x - self.head_baseline[0])
            head_y_offset = head_y - self.head_baseline[1]
            head_y_offset_abs = abs(head_y_offset)
            
            # Head turn
            if head_x_offset > self.head_turn_threshold:
                if head_x_offset >= 0.2:
                    alerts.append(("Severe", "Severe Head Turn"))
                    head_turn_level = 3
                elif head_x_offset >= 0.12:
                    alerts.append(("Moderate", "Moderate Head Turn"))
                    head_turn_level = 2
                else:
                    alerts.append(("Mild", "Mild Head Turn"))
                    head_turn_level = 1
            
            # Head droop
            if head_y_offset_abs > self.head_turn_threshold:
                if head_y_offset < 0:
                    if head_y_offset_abs >= 0.13:
                        alerts.append(("Alert", "Looking Upward"))
                        head_tilt_level = 3
                    elif head_y_offset_abs >= 0.08:
                        alerts.append(("Moderate", "Moderate Looking Upward"))
                        head_tilt_level = 2
                    else:
                        alerts.append(("Mild", "Mild Looking Upward"))
                        head_tilt_level = 1
                else:
                    if head_y_offset_abs >= 0.12:
                        alerts.append(("Severe", "Head drooped"))
                        head_droop_level = 3
                    elif head_y_offset_abs >= 0.07:
                        alerts.append(("Moderate", "Head drooping started"))
                        head_droop_level = 2
                    else:
                        alerts.append(("Mild", "Head drooping symptom"))
                        head_droop_level = 1
            
            # Drowsiness detection
            if yawn_detected and head_droop_level >= 1:
                self.drowsiness_counter += 1
                if self.drowsiness_counter > 5:
                    alerts.append(("Severe", "Severe DROWSINESS Observed"))
                elif self.drowsiness_counter > 3:
                    alerts.append(("Moderate", "Mild DROWSINESS Observed"))
            else:
                self.drowsiness_counter = 0
        
        # Phone detection (Vela-optimized NPU)
        phone_result = self.detect_phone(frame, face_box=(x1, y1, x2, y2))
        if phone_result['likely_mobile_call']:
            alerts.append(("Severe", "Phone Call Detected!"))
        elif phone_result['possible_texting']:
            alerts.append(("Warning", "Possible Texting"))
        
        return {
            'landmarks': landmarks,
            'crop_box': (crop_x1, crop_y1, crop_x2, crop_y2),
            'ear': avg_ear,
            'mar': mar,
            'alerts': alerts
        }
    
    def reset(self):
        """Reset monitoring state"""
        self.eye_closure_counter = 0
        self.blink_counter = 0
        self.yawn_counter = 0
        self.drowsiness_counter = 0
        self.baseline_samples.clear()
        self.baseline_ready = False
        self.ear_history = []
        self.ear_baseline = None


class DMSController:
    """Main controller for DMS on IMX board"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        # System state
        self.state = "DETECTING"
        self.running = False
        
        # Models
        self.face_detector = None
        self.recognizer = None
        self.monitor = None
        self.yolo_worker = None
        
        # Recognition state
        self.current_driver = None
        self.driver_similarity = 0.0
        self.unrecognized_count = 0
        self.unrecognized_threshold = 15
        
        # Threading - smaller queues to reduce memory and processing overhead
        self.frame_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=3)
        self.threads = []
        self.frame_skip_main = 0
        self.frame_skip_main = 0
        
        # UI
        self.active_alerts = defaultdict(float)
        self.alert_duration = 3
        self.fps_deque = deque(maxlen=30)
        
        # MJPEG streaming
        self.mjpeg_server = None
        self.enable_streaming = True  # Enable by default for board
        self.mjpeg_port = 8080
        
        # YOLO configuration
        self.yolo_model_path = "best_640.onnx"
        self.yolo_imgsz = 640
        self.yolo_conf = 0.25
        self.yolo_iou = 0.45
        self.yolo_skip = 10
        self.yolo_frame_counter = 0
        self.draw_boxes = False
        self.class_thresholds = {
            'seatbelt_worn': 0.25,  # BALANCED - allows real detections, filters obvious noise
            'no_seatbelt': 0.20,    # Lower - these detections are usually correct
            'cigarette_hand': 0.30,
            'cigarette_mouth': 0.30,
            'no_cigarette': 0.20
        }
        
        # Seatbelt state
        self.seatbelt_status = None
        self.seatbelt_history = deque(maxlen=20)
        self.seatbelt_vote_threshold = 3  # INCREASED to 3 - stronger consensus needed
        self.seatbelt_position_history = deque(maxlen=10)  # Track detection positions
        self.seatbelt_conf_margin = 0.08
        self.seatbelt_last_emit_ts = 0.0
        self.seatbelt_emit_interval = 0.1
        self.seatbelt_last_seen_ts = 0.0
        self.seatbelt_off_trans_streak = 0
        self.seatbelt_on_trans_streak = 0
        
        # Cigarette state
        self.smoke_status = None
        self.mouth_hold_left = 0
        self.hand_hold_left = 0
        self.cig_hold_frames_mouth = 8
        self.cig_hold_frames_hand = 5
        self.cig_none_min_frames = 2
        self.cig_decay_frames = 15
        self.cig_decay_counter = 0
        self.cig_none_streak = 0
        self.smoke_last_emit_ts = 0.0
        self.smoke_emit_interval = 0.8
        self.cig_face_roi_expand = 0.8
        self.cig_mouth_radius = 0.55
        self.cig_hand_radius = 0.65
        self.cig_hand_to_mouth_dist_norm = 0.22
        self.cig_min_aspect_hand = 1.8
        self.cig_min_aspect_mouth = 1.6
        self.cig_max_area_frac_hand = 0.35
        self.cig_max_area_frac_mouth = 0.30
        self.cig_min_brightness_hand = 110
        self.cig_min_brightness_mouth = 100
        self.require_face_for_cig = True
        self.raw_cig_last_seen_ts = 0.0
        
        print("[DMS-Board] Controller initialized")
    
    def initialize_models(self):
        """Initialize all models with NPU support"""
        print("\n[DMS-Board] Loading models...")
        
        # Face detector (NPU)
        print("  → Loading SCRFD detector (NPU)...")
        self.face_detector = SCRFDDetector("scrfd_500m_full_int8_vela.tflite")
        
        # Face recognizer (NPU)
        print("  → Loading Face Recognition (NPU)...")
        self.recognizer = FaceRecognizer(
            detector_path="scrfd_500m_full_int8_vela.tflite",
            recognizer_path="fr_int8_velaS.tflite",
            database_path="drivers.json"
        )
        
        # Monitoring (CPU)
        print("  → Loading Driver Monitoring (CPU)...")
        self.monitor = DriverMonitoringWrapper()
        
        # YOLO worker (CPU - optional)
        if os.path.isfile(self.yolo_model_path):
            print(f"  → Loading YOLO model (CPU)...")
            try:
                self.yolo_worker = YOLOWorker(
                    model_path=self.yolo_model_path,
                    imgsz=self.yolo_imgsz,
                    conf=self.yolo_conf,
                    iou=self.yolo_iou,
                    class_thresholds=self.class_thresholds,
                    names=DEFAULT_NAMES,
                    nms_topk=300,
                    sleep_interval=15.0  # Increased from 10s to 15s
                )
                self.yolo_worker.start()
                time.sleep(0.5)  # Give thread time to start
                if self.yolo_worker.is_alive():
                    print(f"  → YOLO worker thread is running")
                else:
                    print(f"  → WARNING: YOLO worker thread failed to start!")
                    self.yolo_worker = None
            except Exception as e:
                print(f"  → YOLO worker failed to initialize: {e}")
                self.yolo_worker = None
        else:
            print(f"  → YOLO model not found: {self.yolo_model_path} (skipping)")
        
        print("[DMS-Board] All models loaded\n")
    
    def detection_thread(self):
        """Face detection thread"""
        print("[Thread] Face detection started")
        frame_count = 0
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.2)  # Increased timeout
                
                # Skip every other frame when in MONITORING mode to reduce NPU load
                frame_count += 1
                if self.state == "MONITORING" and frame_count % 2 != 0:
                    continue
                
                detections = self.face_detector.detect(frame, conf_threshold=0.70)
                
                if detections:
                    det = detections[0]
                    self.detection_queue.put({
                        'box': det['box'],
                        'landmarks': det['landmarks'],
                        'score': det['score'],
                        'frame': frame.copy()
                    })
                else:
                    self.detection_queue.put(None)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Detection] Error: {e}")
        
        print("[Thread] Face detection stopped")
    
    def recognition_thread(self):
        """Face recognition thread"""
        print("[Thread] Face recognition started")
        
        no_face_count = 0
        face_visible_count = 0
        
        while self.running and self.state == "RECOGNIZING":
            try:
                detection = self.detection_queue.get(timeout=0.2)
                
                if detection is None:
                    # No face detected in frame
                    no_face_count += 1
                    face_visible_count = 0
                    self.unrecognized_count = 0
                    
                    if no_face_count >= 10:  # Show message after ~10 frames
                        print("\r[Recognition] No face detected - please look at camera", end='', flush=True)
                    continue
                
                # Face detected - reset no face counter
                no_face_count = 0
                face_visible_count += 1
                
                if face_visible_count == 1:
                    print("\n[Recognition] Face detected - recognizing...")
                
                frame = detection['frame']
                
                # Recognize
                result = self.recognizer.recognize_face(frame)
                
                if result and result['driver_id'] is not None:
                    self.result_queue.put({
                        'type': 'recognized',
                        'name': result['name'],
                        'similarity': result['similarity']
                    })
                    break
                else:
                    # Face visible but not recognized
                    self.unrecognized_count += 1
                    if self.unrecognized_count > self.unrecognized_threshold:
                        print(f"\n[Recognition] Face visible but not recognized after {self.unrecognized_count} frames")
                        self.result_queue.put({'type': 'enroll_request'})
                        break
                    
            except queue.Empty:
                time.sleep(0.02)  # Reduce busy-waiting CPU usage
                continue
            except Exception as e:
                print(f"[Recognition] Error: {e}")
        
        print("[Thread] Face recognition stopped")
    
    def monitoring_thread(self):
        """Driver monitoring thread"""
        print("[Thread] Driver monitoring started")
        
        self.monitor.reset()
        no_face_count = 0
        no_face_threshold = 30  # Reset after ~30 frames without face (~1 second)
        
        while self.running and self.state == "MONITORING":
            try:
                detection = self.detection_queue.get(timeout=0.15)
                
                if detection is None:
                    # No face detected
                    no_face_count += 1
                    
                    if no_face_count >= no_face_threshold:
                        print("\n[Monitoring] Face lost - returning to recognition mode")
                        self.current_driver = None
                        self.driver_similarity = 0.0
                        self.state = "RECOGNIZING"
                        
                        # Start new recognition thread
                        recog_thread = threading.Thread(target=self.recognition_thread, daemon=True)
                        recog_thread.start()
                        self.threads.append(recog_thread)
                        break
                    
                    if no_face_count % 10 == 0:
                        print(f"[Monitoring] No face detected for {no_face_count} frames...")
                    continue
                
                # Face detected - reset counter
                no_face_count = 0
                
                frame = detection['frame']
                box = list(map(int, detection['box']))
                
                # Submit to YOLO worker (no copy needed, it stores reference)
                if self.yolo_worker is not None:
                    self.yolo_worker.submit(frame)
                
                result = self.monitor.analyze(frame, box)
                
                if result:
                    # Get YOLO detections
                    yolo_dets = []
                    if self.yolo_worker is not None:
                        yolo_dets = self.yolo_worker.get_latest()
                    
                    # Process seatbelt and cigarette
                    self._process_seatbelt(frame, box, yolo_dets)
                    self._process_cigarette(frame, box, yolo_dets, result['landmarks'])
                    
                    result['yolo_dets'] = yolo_dets
                    result['seatbelt_status'] = self.seatbelt_status
                    result['smoke_status'] = self.smoke_status
                    
                    self.result_queue.put({
                        'type': 'monitoring',
                        'data': result,
                        'detection': detection
                    })
                    
            except queue.Empty:
                time.sleep(0.02)  # Reduce busy-waiting CPU usage
                continue
            except Exception as e:
                print(f"[Monitoring] Error: {e}")
        
        print("[Thread] Driver monitoring stopped")
    
    def trigger_enrollment(self):
        """Trigger enrollment"""
        print("\n" + "="*70)
        print("UNKNOWN DRIVER - ENROLLMENT REQUIRED")
        print("="*70)
        print("Run: python3 enroll_industrial_board.py --id <ID> --name <NAME>")
        print("="*70 + "\n")
        self.state = "ENROLLING"
    
    def start(self):
        """Start DMS system"""
        print("\n" + "="*70)
        print("DRIVER MONITORING SYSTEM - IMX BOARD")
        print("="*70)
        print("\nControls:")
        print("  Ctrl+C - Quit")
        print("  View stream in VLC: http://<board-ip>:8080/video")
        print("="*70 + "\n")
        
        self.initialize_models()
        
        # Start MJPEG streaming server
        if self.enable_streaming:
            self.mjpeg_server = MJPEGServer(host='0.0.0.0', port=self.mjpeg_port)
            if self.mjpeg_server.start():
                print(f"[DMS-Board] Stream ready at port {self.mjpeg_port}\n")
            else:
                self.mjpeg_server = None
        
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return
        
        self.running = True
        
        # Start threads
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()
        self.threads.append(detection_thread)
        
        self.state = "RECOGNIZING"
        recog_thread = threading.Thread(target=self.recognition_thread, daemon=True)
        recog_thread.start()
        self.threads.append(recog_thread)
        
        print("[DMS-Board] System running\n")
        
        last_detection = None
        no_face_frames = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Skip frames to reduce CPU (process ~10 fps instead of 30)
                self.frame_skip_main += 1
                if self.frame_skip_main % 3 == 0:  # Process every 3rd frame (66% reduction)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    else:
                        time.sleep(0.002)  # Queue full, yield CPU
                
                # Process results (limit processing rate)
                result_count = 0
                while not self.result_queue.empty() and result_count < 2:
                    try:
                        result = self.result_queue.get_nowait()
                        result_count += 1
                        
                        if result['type'] == 'recognized':
                            self.current_driver = result['name']
                            self.state = "MONITORING"
                            
                            print(f"\n✓ DRIVER: {self.current_driver}\n")
                            
                            monitor_thread = threading.Thread(target=self.monitoring_thread, daemon=True)
                            monitor_thread.start()
                            self.threads.append(monitor_thread)
                        
                        elif result['type'] == 'monitoring':
                            last_detection = result
                            no_face_frames = 0
                        
                        elif result['type'] == 'enroll_request':
                            self.trigger_enrollment()
                    
                    except queue.Empty:
                        break
                
                # Clear last_detection if in RECOGNIZING mode for better UI feedback
                if self.state == "RECOGNIZING":
                    no_face_frames += 1
                    if no_face_frames > 5:  # Clear after 5 frames
                        last_detection = None
                
                # Draw UI
                display_frame = frame.copy()
                self.draw_ui(display_frame, last_detection)
                
                # Draw YOLO boxes if enabled and detections available
                if self.draw_boxes and last_detection and 'data' in last_detection:
                    yolo_dets = last_detection['data'].get('yolo_dets', [])
                    for det in yolo_dets:
                        x1, y1, x2, y2 = det['box']
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                        label = f"{det['name']}: {det['conf']:.2f}"
                        cv2.putText(display_frame, label, (int(x1), int(y1)-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Update MJPEG stream
                if self.mjpeg_server is not None:
                    self.mjpeg_server.update_frame(display_frame)
                
                # FPS
                self.fps_deque.append(time.time())
                if len(self.fps_deque) > 1:
                    fps = len(self.fps_deque) / (self.fps_deque[-1] - self.fps_deque[0])
                    # Print FPS to console instead of display
                    if int(time.time()) % 5 == 0:  # Every 5 seconds
                        print(f"[DMS-Board] FPS: {fps:.1f}")
                
                # Skip GUI on embedded board - comment out cv2.imshow
                # cv2.imshow("DMS - IMX Board", display_frame)
                
                # Minimal wait for key input with CPU yield
                key = cv2.waitKey(10) & 0xFF  # Increased to 10ms for less CPU usage
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_system()
                
                time.sleep(0.02)  # Additional yield to reduce CPU usage (increased from 0.01)
                
                time.sleep(0.01)  # Additional yield to reduce CPU usage
        
        finally:
            self.running = False
            
            if self.yolo_worker is not None:
                self.yolo_worker.stop()
            
            if self.mjpeg_server is not None:
                self.mjpeg_server.stop()
            
            cap.release()
            # cv2.destroyAllWindows()  # Skip on embedded board
            
            for thread in self.threads:
                thread.join(timeout=1.0)
            
            print("\n[DMS-Board] System stopped")
    
    def _process_seatbelt(self, frame, face_box, yolo_dets):
        """Process seatbelt detections with voting logic"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_box
        face_h = y2 - y1
        
        # Debug: show all YOLO detections
        if yolo_dets:
            belt_cig_dets = [d for d in yolo_dets if _norm_name(d['name']) in ['seatbelt_worn', 'no_seatbelt', 'cigarette_hand', 'cigarette_mouth', 'no_cigarette']]
            if belt_cig_dets:
                print(f"[YOLO-Raw] Found {len(belt_cig_dets)} belt/cig detections: {[f"{d['name']}({d['conf']:.2f})" for d in belt_cig_dets]}")
        
        # Build LARGER chest ROI covering actual torso where seatbelt appears
        # Seatbelt goes diagonally across chest from shoulder to opposite hip
        face_w = x2 - x1
        face_cx = (x1 + x2) // 2  # Face center X
        
        # Horizontal: Wide coverage centered on face (±40% of frame width)
        torso_half_width = int(w * 0.4)
        chest_x1 = max(0, face_cx - torso_half_width)
        chest_x2 = min(w, face_cx + torso_half_width)
        
        # Vertical: Start from NECK/SHOULDERS (just below face), extend to waist
        # Seatbelt appears from shoulders down to hip area
        chest_y1 = max(0, int(y2 - face_h * 0.2))  # Start at neck/shoulder (20% up from chin)
        chest_y2 = min(h, int(y2 + face_h * 2.5))  # Extend to waist/hip area
        
        print(f"[Belt] Chest ROI: y:{chest_y1}-{chest_y2}, x:{chest_x1}-{chest_x2} | Face: ({x1},{y1})-({x2},{y2})")
        
        # Find seatbelt detections
        seatbelt_worn_hit = False
        no_seatbelt_hit = False
        seatbelt_worn_conf = 0.0
        no_seatbelt_conf = 0.0
        
        current_time = time.time()
        
        # Helper: calculate IoU between two boxes
        def box_iou(box1, box2):
            x1_max = max(box1[0], box2[0])
            y1_max = max(box1[1], box2[1])
            x2_min = min(box1[2], box2[2])
            y2_min = min(box1[3], box2[3])
            inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        
        belt_dets_found = 0
        for det in yolo_dets:
            name_norm = _norm_name(det['name'])
            
            if name_norm not in ['seatbelt_worn', 'no_seatbelt']:
                continue
            
            belt_dets_found += 1
            
            bx1, by1, bx2, by2 = det['box']
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            
            # Accept if center in chest ROI OR has any reasonable overlap (more permissive for large ROI)
            in_chest = (chest_x1 <= bcx <= chest_x2) and (chest_y1 <= bcy <= chest_y2)
            iou = box_iou([bx1, by1, bx2, by2], [chest_x1, chest_y1, chest_x2, chest_y2])
            
            # Accept if center is inside OR has 5% overlap (relaxed for stability)
            if not in_chest and iou < 0.05:
                print(f"[Belt-Debug] ✗ {det['name']} outside torso: center=({bcx:.0f},{bcy:.0f}), IoU={iou:.2f}")
                continue
            
            # BALANCED GEOMETRIC VALIDATION
            # Vertical position: Allow chest area (not extreme top/bottom)
            y_frac = bcy / h
            if y_frac < 0.15:  # Relaxed to 15% (was 20%)
                print(f"[Belt-Debug] ✗ {det['name']} too high: y_frac={y_frac:.2f}")
                continue
            
            # Aspect ratio: Seatbelt is somewhat elongated (relaxed)
            bw = bx2 - bx1
            bh = by2 - by1
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect < 1.2:  # Relaxed to 1.2 (was 1.3)
                print(f"[Belt-Debug] ✗ {det['name']} too square: aspect={aspect:.2f}")
                continue
            
            # Size validation: More permissive range
            box_area = bw * bh
            frame_area = w * h
            area_frac = box_area / frame_area
            if area_frac < 0.003:  # Relaxed to 0.3% (was 0.5%)
                print(f"[Belt-Debug] ✗ {det['name']} too small: area={area_frac*100:.2f}%")
                continue
            if area_frac > 0.4:  # Relaxed to 40% (was 30%)
                print(f"[Belt-Debug] ✗ {det['name']} too large: area={area_frac*100:.2f}%")
                continue
            
            if name_norm == 'seatbelt_worn':
                seatbelt_worn_hit = True
                seatbelt_worn_conf = max(seatbelt_worn_conf, det['conf'])
                print(f"[Belt-Debug] ✓ seatbelt_worn ACCEPTED: conf={det['conf']:.2f}, center=({bcx:.0f},{bcy:.0f})")
            elif name_norm == 'no_seatbelt':
                no_seatbelt_hit = True
                no_seatbelt_conf = max(no_seatbelt_conf, det['conf'])
                print(f"[Belt-Debug] ✓ no_seatbelt ACCEPTED: conf={det['conf']:.2f}, center=({bcx:.0f},{bcy:.0f})")
        
        if belt_dets_found == 0:
            print(f"[Belt] No seatbelt detections in YOLO results")
        
        # Count how many ACCEPTED detections (not all YOLO detections)
        worn_count = sum(1 for d in yolo_dets if _norm_name(d['name']) == 'seatbelt_worn')
        no_belt_count = sum(1 for d in yolo_dets if _norm_name(d['name']) == 'no_seatbelt')
        
        # BALANCED: Require minimum 3 detections (reduced from 5 for better detection)
        MIN_DETECTION_COUNT = 3
        if seatbelt_worn_hit and worn_count < MIN_DETECTION_COUNT:
            print(f"[Belt] Rejecting seatbelt_worn: only {worn_count} detections (need {MIN_DETECTION_COUNT})")
            seatbelt_worn_hit = False
        if no_seatbelt_hit and no_belt_count < MIN_DETECTION_COUNT:
            print(f"[Belt] Rejecting no_seatbelt: only {no_belt_count} detections (need {MIN_DETECTION_COUNT})")
            no_seatbelt_hit = False
        
        if seatbelt_worn_hit and no_seatbelt_hit:
            # First check confidence difference
            conf_diff = abs(seatbelt_worn_conf - no_seatbelt_conf)
            if conf_diff >= self.seatbelt_conf_margin:
                # Clear winner by confidence
                if seatbelt_worn_conf > no_seatbelt_conf:
                    no_seatbelt_hit = False
                    print(f"[Belt] Conflict: seatbelt_worn wins by confidence ({seatbelt_worn_conf:.2f} > {no_seatbelt_conf:.2f})")
                else:
                    seatbelt_worn_hit = False
                    print(f"[Belt] Conflict: no_seatbelt wins by confidence ({no_seatbelt_conf:.2f} > {seatbelt_worn_conf:.2f})")
            else:
                # Confidences too close - use COUNT as tiebreaker
                if worn_count > no_belt_count:
                    no_seatbelt_hit = False
                    print(f"[Belt] Conflict: seatbelt_worn wins by count ({worn_count} vs {no_belt_count})")
                elif no_belt_count > worn_count:
                    seatbelt_worn_hit = False
                    print(f"[Belt] Conflict: no_seatbelt wins by count ({no_belt_count} vs {worn_count})")
                else:
                    # Exact tie - reject both (rare case)
                    print(f"[Belt] Exact tie - rejecting both (conf={seatbelt_worn_conf:.2f}, count={worn_count})")
                    seatbelt_worn_hit = False
                    no_seatbelt_hit = False
        
        # Update history
        if seatbelt_worn_hit:
            self.seatbelt_history.append(1)
            self.seatbelt_last_seen_ts = current_time
        elif no_seatbelt_hit:
            self.seatbelt_history.append(-1)
            self.seatbelt_last_seen_ts = current_time
        else:
            self.seatbelt_history.append(0)
        
        # Check if no detections for 5 frames
        if current_time - self.seatbelt_last_seen_ts > 0.15:
            if len(self.seatbelt_history) > 0:
                recent = list(self.seatbelt_history)[-5:]
                if all(v == 0 for v in recent):
                    self.seatbelt_status = None
                    return
        
        # Voting
        if len(self.seatbelt_history) > 0:
            vote_sum = sum(self.seatbelt_history)
            
            if abs(vote_sum) >= self.seatbelt_vote_threshold:
                print(f"[Belt] Vote sum: {vote_sum}/{self.seatbelt_vote_threshold}, history={list(self.seatbelt_history)[-10:]}")
            
            # Quick transitions
            if seatbelt_worn_hit:
                self.seatbelt_on_trans_streak += 1
                self.seatbelt_off_trans_streak = 0
                if self.seatbelt_on_trans_streak >= 1 and self.seatbelt_status == 'no':
                    self.seatbelt_status = 'worn'
                    print("[Belt] Seatbelt Worn")
                    self.seatbelt_last_emit_ts = current_time
            elif no_seatbelt_hit:
                self.seatbelt_off_trans_streak += 1
                self.seatbelt_on_trans_streak = 0
                if self.seatbelt_off_trans_streak >= 1 and self.seatbelt_status == 'worn':
                    self.seatbelt_status = 'no'
                    print("[Belt] NO SEATBELT")
                    self.seatbelt_last_emit_ts = current_time
            else:
                self.seatbelt_on_trans_streak = 0
                self.seatbelt_off_trans_streak = 0
            
            # Threshold-based status
            if vote_sum >= self.seatbelt_vote_threshold:
                if self.seatbelt_status != 'worn':
                    self.seatbelt_status = 'worn'
                    print(f"[Belt] Status changed to: WORN (vote_sum={vote_sum})")
                    self.seatbelt_last_emit_ts = current_time
            elif vote_sum <= -self.seatbelt_vote_threshold:
                if self.seatbelt_status != 'no':
                    self.seatbelt_status = 'no'
                    print(f"[Belt] Status changed to: NO SEATBELT (vote_sum={vote_sum})")
                    self.seatbelt_last_emit_ts = current_time
        
        # Periodic re-emit
        if self.seatbelt_status and (current_time - self.seatbelt_last_emit_ts) >= self.seatbelt_emit_interval:
            if self.seatbelt_status == 'worn':
                print("[Belt] Seatbelt Worn")
            else:
                print("[Belt] NO SEATBELT")
            self.seatbelt_last_emit_ts = current_time
    
    def _get_mouth_roi(self, landmarks, face_box, frame_shape, expand_ratio=0.20):
        """Extract mouth ROI from FaceMesh landmarks"""
        try:
            x1, y1, x2, y2 = face_box
            face_w = x2 - x1
            face_h = y2 - y1
            h, w = frame_shape[:2]
            
            # Get mouth corner landmarks (61=left, 291=right, 13=top, 14=bottom)
            mouth_top = landmarks[13, :2]
            mouth_bottom = landmarks[14, :2]
            mouth_left = landmarks[61, :2]
            mouth_right = landmarks[291, :2]
            
            # Convert to absolute coordinates
            mouth_points = np.array([mouth_top, mouth_bottom, mouth_left, mouth_right])
            mouth_abs = mouth_points * [face_w, face_h] + [x1, y1]
            
            # Calculate bounding box with expansion
            mx1 = int(mouth_abs[:, 0].min() - face_w * expand_ratio)
            my1 = int(mouth_abs[:, 1].min() - face_h * expand_ratio)
            mx2 = int(mouth_abs[:, 0].max() + face_w * expand_ratio)
            my2 = int(mouth_abs[:, 1].max() + face_h * expand_ratio)
            
            # Clamp to frame boundaries
            mx1 = max(0, mx1)
            my1 = max(0, my1)
            mx2 = min(w, mx2)
            my2 = min(h, my2)
            
            # Return ROI and mouth center
            mouth_cx = (mx1 + mx2) / 2
            mouth_cy = (my1 + my2) / 2
            
            return (mx1, my1, mx2, my2, mouth_cx, mouth_cy)
        
        except Exception as e:
            return None
    
    def _process_cigarette(self, frame, face_box, yolo_dets, landmarks):
        """Process cigarette detections with gating and logic"""
        if landmarks is None:
            return
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_box
        face_w = x2 - x1
        face_h = y2 - y1
        face_area = face_w * face_h
        
        # Extract PRECISE mouth ROI from landmarks
        mouth_roi = self._get_mouth_roi(landmarks, face_box, frame.shape, expand_ratio=0.20)
        
        if mouth_roi is None:
            return
        
        mx1, my1, mx2, my2, mouth_cx, mouth_cy = mouth_roi
        mouth_w = mx2 - mx1
        mouth_h = my2 - my1
        
        print(f"[Cig-ROI] Mouth ROI: ({mx1},{my1})-({mx2},{my2}), center=({int(mouth_cx)},{int(mouth_cy)}), size={mouth_w}x{mouth_h}")
        
        # LARGER face ROI to handle landmark flickering - cigarettes held near face/mouth
        # Use frame-relative expansion for stability
        face_cx = (x1 + x2) // 2
        face_cy = (y1 + y2) // 2
        
        # Expand to ±40% of frame width/height from face center (larger, more stable)
        roi_half_w = int(w * 0.4)
        roi_half_h = int(h * 0.4)
        roi_x1 = max(0, face_cx - roi_half_w)
        roi_y1 = max(0, face_cy - roi_half_h)
        roi_x2 = min(w, face_cx + roi_half_w)
        roi_y2 = min(h, face_cy + roi_half_h)
        
        current_time = time.time()
        
        # Find cigarette detections (only log actionable ones)
        mouth_cig_hit = False
        hand_cig_hit = False
        raw_cig_seen = False
        cig_det_count = 0
        
        for det in yolo_dets:
            name_norm = _norm_name(det['name'])
            
            # Track if any cigarette class seen
            if name_norm in ['cigarette_hand', 'cigarette_mouth', 'no_cigarette']:
                raw_cig_seen = True
                if name_norm in ['cigarette_hand', 'cigarette_mouth']:
                    cig_det_count += 1
            
            if name_norm not in ['cigarette_hand', 'cigarette_mouth']:
                continue
            
            bx1, by1, bx2, by2 = det['box']
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            
            # Check if detection is IN mouth ROI
            in_mouth_roi = (mx1 <= bcx <= mx2) and (my1 <= bcy <= my2)
            
            # Distance to mouth center
            dist_to_mouth = np.sqrt((bcx - mouth_cx)**2 + (bcy - mouth_cy)**2)
            dist_norm = dist_to_mouth / mouth_h  # Normalize by mouth height
            
            print(f"[Cig-ROI] {det['name']}: center=({int(bcx)},{int(bcy)}), dist_to_mouth={int(dist_to_mouth)}px ({dist_norm:.2f}× mouth_h), in_mouth_roi={in_mouth_roi}, conf={det['conf']:.2f}")
            
            # CIGARETTE_MOUTH: MUST be in mouth ROI or very close
            if name_norm == 'cigarette_mouth':
                if in_mouth_roi or dist_norm < 0.8:  # In ROI or within 80% of mouth height
                    print(f"[Cig-ROI] ✓ CIGARETTE_MOUTH ACCEPTED (in_roi={in_mouth_roi}, dist={dist_norm:.2f})")
                    mouth_cig_hit = True
                else:
                    print(f"[Cig-ROI] ✗ CIGARETTE_MOUTH REJECTED (too far from mouth, dist={dist_norm:.2f})")
            
            # CIGARETTE_HAND: Can be near face but not necessarily in mouth
            elif name_norm == 'cigarette_hand':
                # Check if in larger face ROI (±40% frame)
                in_face_roi = (roi_x1 <= bcx <= roi_x2) and (roi_y1 <= bcy <= roi_y2)
                
                # If hand-held cigarette very close to mouth, escalate to mouth
                if dist_norm < 0.5:  # Within 50% of mouth height
                    print(f"[Cig-ROI] ✓ CIGARETTE_HAND NEAR MOUTH - Escalating to mouth (dist={dist_norm:.2f})")
                    mouth_cig_hit = True
                elif in_face_roi:
                    print(f"[Cig-ROI] ✓ CIGARETTE_HAND in face ROI (dist={dist_norm:.2f})")
                    hand_cig_hit = True
                else:
                    print(f"[Cig-ROI] ✗ CIGARETTE_HAND outside face ROI")
                print(f"[Cig] ✓ CIGARETTE IN HAND detected!")
                # Escalate hand to mouth if very close
                if dist_norm <= 0.30:
                    mouth_cig_hit = True
                    hand_cig_hit = False
                    print(f"[Cig] ↑ Escalated to MOUTH (close to lips)")
        
        if raw_cig_seen:
            self.raw_cig_last_seen_ts = current_time
        
        # Update holds
        if mouth_cig_hit:
            self.mouth_hold_left = self.cig_hold_frames_mouth
            self.hand_hold_left = 0
            self.cig_decay_counter = 0
            self.cig_none_streak = 0
        elif hand_cig_hit:
            self.hand_hold_left = self.cig_hold_frames_hand
            self.mouth_hold_left = 0
            self.cig_decay_counter = 0
            self.cig_none_streak = 0
        else:
            if self.mouth_hold_left > 0:
                self.mouth_hold_left -= 1
            if self.hand_hold_left > 0:
                self.hand_hold_left -= 1
            else:
                self.cig_decay_counter += 1
        
        # Determine status
        prev_status = self.smoke_status
        
        if self.mouth_hold_left > 0:
            self.smoke_status = 'mouth'
        elif self.hand_hold_left > 0:
            self.smoke_status = 'hand'
        elif self.cig_decay_counter >= self.cig_decay_frames:
            self.cig_none_streak += 1
            if self.cig_none_streak >= self.cig_none_min_frames:
                self.smoke_status = 'none'
        
        # Emit alerts
        if self.smoke_status != prev_status or (current_time - self.smoke_last_emit_ts) >= self.smoke_emit_interval:
            if self.smoke_status == 'mouth':
                print("[Smoke] Cigarette in Mouth")
                self.smoke_last_emit_ts = current_time
            elif self.smoke_status == 'hand':
                print("[Smoke] Cigarette in Hand")
                self.smoke_last_emit_ts = current_time
            elif self.smoke_status == 'none':
                # Suppress if raw cig seen recently
                if current_time - self.raw_cig_last_seen_ts > 0.5:
                    print("[Smoke] No Cigarette")
                    self.smoke_last_emit_ts = current_time
    
    def draw_ui(self, frame, last_detection):
        """Draw UI"""
        h, w = frame.shape[:2]
        
        # Status
        status_text = f"{self.state}"
        if self.current_driver:
            status_text += f" | {self.current_driver}"
        
        cv2.putText(frame, status_text, (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show "No Face Detected" message when in RECOGNIZING mode with no detection
        if self.state == "RECOGNIZING" and last_detection is None:
            cv2.putText(frame, "NO FACE DETECTED", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "Please look at camera", (w//2 - 120, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Monitoring alerts
        if last_detection and self.state == "MONITORING":
            det = last_detection['detection']
            data = last_detection['data']
            
            box = list(map(int, det['box']))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Update alerts
            current_time = time.time()
            for severity, msg in data.get('alerts', []):
                self.active_alerts[msg] = current_time
            
            # Expire old
            expired = [k for k, t in self.active_alerts.items() 
                      if current_time - t > self.alert_duration]
            for k in expired:
                del self.active_alerts[k]
            
            # Draw alerts
            alert_y = 60
            for alert_msg in list(self.active_alerts.keys()):
                if "Severe" in alert_msg:
                    color = (0, 0, 255)  # Red
                elif "Moderate" in alert_msg:
                    color = (0, 255, 255)  # Yellow
                elif "Warning" in alert_msg or "Alert" in alert_msg:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow (changed from white)
                
                cv2.putText(frame, alert_msg, (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                alert_y += 20
            
            # Draw seatbelt status (always show when monitoring)
            if self.seatbelt_status == 'worn':
                cv2.putText(frame, "[BELT] WORN", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                alert_y += 25
            elif self.seatbelt_status == 'no':
                cv2.putText(frame, "[BELT] NOT WORN - ALERT!", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                alert_y += 25
            elif self.seatbelt_status is None:
                cv2.putText(frame, "[BELT] Detecting...", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                alert_y += 25
            
            # Draw cigarette status (always show when monitoring)
            if self.smoke_status == 'mouth':
                cv2.putText(frame, "[SMOKE] IN MOUTH - ALERT!", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                alert_y += 25
            elif self.smoke_status == 'hand':
                cv2.putText(frame, "[SMOKE] IN HAND - WARNING!", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                alert_y += 25
            elif self.smoke_status == 'none':
                cv2.putText(frame, "[SMOKE] Clear", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                alert_y += 25
            else:
                cv2.putText(frame, "[SMOKE] Detecting...", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                alert_y += 25
            
            # Show YOLO detection count
            yolo_dets = data.get('yolo_dets', [])
            if yolo_dets:
                yolo_count_text = f"YOLO: {len(yolo_dets)} detections"
                cv2.putText(frame, yolo_count_text, (10, h-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def reset_system(self):
        """Reset to recognition mode"""
        print("\n[DMS-Board] Resetting...\n")
        self.state = "RECOGNIZING"
        self.current_driver = None
        self.unrecognized_count = 0
        self.active_alerts.clear()
        
        # Reset YOLO state
        self.seatbelt_status = None
        self.seatbelt_history.clear()
        self.smoke_status = None
        self.mouth_hold_left = 0
        self.hand_hold_left = 0
        self.cig_decay_counter = 0


if __name__ == "__main__":
    controller = DMSController(camera_id=0)
    controller.start()
