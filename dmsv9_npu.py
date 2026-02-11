import cv2
import time
import argparse
import numpy as np
from collections import deque
from datetime import datetime
from collections import defaultdict
import threading
import http.server
import socketserver
from io import BytesIO
import glob
import os

# TFLite runtime (board uses tflite_runtime, PC uses tensorflow.lite)
try:
    import tflite_runtime.interpreter as tflite
    print("[Runtime] Using tflite_runtime")
except ImportError:
    import tensorflow.lite as tflite
    print("[Runtime] Using tensorflow.lite")


# ============================================================
# MJPEG Streaming Server for VLC viewing
# ============================================================

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
                        ret, jpeg = cv2.imencode('.jpg', self.server.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        if ret:
                            self.wfile.write(b"--jpgboundary\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b"\r\n")
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass


class MJPEGServer:
    """MJPEG streaming server for network viewing"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.current_frame = None
    
    def start(self):
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
        if self.server is not None:
            self.server.current_frame = frame
    
    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            print("[MJPEG] Server stopped")


# ============================================================
# NPU TFLite Model Loader + Dequantization Helper
# ============================================================

def dequantize_tensor(interpreter, tensor_index, output_detail):
    """
    Get a tensor from the interpreter and dequantize if needed.
    For INT8/UINT8 quantized outputs: float = (int_val - zero_point) * scale
    For float32 outputs: returns as-is.
    """
    raw = interpreter.get_tensor(tensor_index)
    if raw.dtype == np.float32:
        return raw
    # INT8 or UINT8 — need manual dequantization
    qp = output_detail.get('quantization_parameters', {})
    scales = qp.get('scales', None)
    zero_points = qp.get('zero_points', None)
    if scales is not None and len(scales) > 0:
        scale = scales[0]
        zp = zero_points[0] if zero_points is not None and len(zero_points) > 0 else 0
    else:
        # Fallback to legacy 'quantization' tuple (scale, zero_point)
        legacy = output_detail.get('quantization', (0.0, 0))
        scale = legacy[0] if legacy[0] != 0.0 else 1.0
        zp = legacy[1]
    return (raw.astype(np.float32) - zp) * scale


def load_npu_model(model_path):
    """Load a Vela-compiled TFLite model on the Ethos-U NPU"""
    try:
        ethosu_delegate = tflite.load_delegate(
            "/usr/lib/libethosu_delegate.so",
            {
                "device_name": "/dev/ethosu0",
                "cache_file_path": ".",
                "enable_cycle_counter": "false",
            }
        )
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[ethosu_delegate]
        )
        print(f"[NPU] Loaded {model_path} with Ethos-U delegate")
    except Exception as e:
        print(f"[NPU] Ethos-U delegate failed ({e}), falling back to CPU")
        interpreter = tflite.Interpreter(model_path=model_path)
    
    interpreter.allocate_tensors()
    return interpreter


# ============================================================
# Face Landmark Model Wrapper (replaces MediaPipe FaceMesh)
# ============================================================

class FaceLandmarkDetector:
    """
    Face Landmark detector using NXP face_landmark_ptq_vela.tflite
    Input:  [1, 192, 192, 3] float32 (0.0 - 1.0 normalized)
    Output0: [1, 1, 1, 1] float32 - face confidence
    Output1: [1, 1, 1, 1404] float32 - 468 landmarks x 3 (x, y, z)
    """
    
    def __init__(self, model_path="face_landmark_ptq_vela.tflite"):
        self.interpreter = load_npu_model(model_path)
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details['shape']  # [1, 192, 192, 3]
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        
        # NXP convention: scores at output[0], landmarks at output[1]
        self.score_index = self.output_details[0]['index']
        self.landmark_index = self.output_details[1]['index']
        
        print(f"[FaceLandmark] Input: {self.input_shape}, dtype={self.input_details['dtype']}")
        for i, od in enumerate(self.output_details):
            print(f"[FaceLandmark] Output {i}: {od['shape']}, dtype={od['dtype']}")
    
    def predict(self, face_crop_bgr):
        """
        Run face landmark inference on a cropped face image.
        
        Args:
            face_crop_bgr: BGR face crop (any size, will be resized)
        
        Returns:
            confidence: float (0-1)
            landmarks: numpy array [468, 3] normalized (x, y, z) relative to crop
        """
        # Preprocess: resize, BGR->RGB, normalize to [-1, 1] (NXP convention)
        img = cv2.resize(face_crop_bgr, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 128.0) / 128.0
        img = np.expand_dims(img, axis=0)  # [1, 192, 192, 3]
        
        # Run inference
        self.interpreter.set_tensor(self.input_details['index'], img)
        self.interpreter.invoke()
        
        # Parse outputs — NXP convention: scores at [0], landmarks at [1]
        confidence = float(self.interpreter.get_tensor(self.score_index).flatten()[0])
        raw_landmarks = self.interpreter.get_tensor(self.landmark_index).flatten().astype(np.float32)
        
        if raw_landmarks.size < 1404:
            return 0.0, None
        
        # Reshape to [468, 3] — x, y, z
        # NXP convention: divide by model input dimensions to normalize to 0-1
        landmarks = raw_landmarks[:1404].reshape(468, 3)
        landmarks[:, 0] /= self.input_w   # x -> 0-1
        landmarks[:, 1] /= self.input_h   # y -> 0-1
        # z is relative depth, normalize by width (NXP convention)
        landmarks[:, 2] /= self.input_w
        
        return confidence, landmarks


# ============================================================
# Iris Landmark Model Wrapper (replaces MediaPipe iris refinement)
# ============================================================

class IrisLandmarkDetector:
    """
    Iris/Eye Landmark detector using NXP iris_landmark_ptq_vela.tflite
    Input:  [1, 64, 64, 3] float32 normalized to [-1, 1]
    Output0: eye contour landmarks (71 points x 3)
    Output1: iris landmarks (5 points x 3)
    
    Preprocessing: (pixel - 128) / 128.0 -> range [-1, 1]
    Right eye: input is flipped horizontally, output x-coords inverted
    (Matches NXP reference implementation exactly)
    """
    
    # Eye contour connection topology for drawing
    EYE_LANDMARK_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
        (0, 9), (8, 14),
    ]
    
    # Eye ROI landmarks (from 468 face landmarks)
    LEFT_EYE_START = 33
    LEFT_EYE_END = 133
    RIGHT_EYE_START = 362
    RIGHT_EYE_END = 263
    ROI_SCALE = 2
    
    def __init__(self, model_path="iris_landmark_ptq_vela.tflite"):
        self.interpreter = load_npu_model(model_path)
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details['shape']
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        
        # NXP convention: eye contour at output[0], iris at output[1]
        self.eye_index = self.output_details[0]['index']
        self.iris_index = self.output_details[1]['index']
        
        print(f"[IrisLandmark] Input: {self.input_shape}, dtype={self.input_details['dtype']}")
        for i, od in enumerate(self.output_details):
            print(f"[IrisLandmark] Output {i}: {od['shape']}, dtype={od['dtype']}")
    
    def get_eye_roi(self, face_landmarks, side):
        """
        Get left/right eye ROI from 468 face landmarks (NXP method).
        Args:
            face_landmarks: list of LandmarkPoint objects (468 points)
            side: 0=left eye, 1=right eye
            w, h: frame dimensions (implicit via landmark coords)
        Returns: (xmin, ymin, xmax, ymax) in pixel coordinates
        """
        if side == 0:
            start_idx = self.LEFT_EYE_START
            end_idx = self.LEFT_EYE_END
        else:
            start_idx = self.RIGHT_EYE_START
            end_idx = self.RIGHT_EYE_END
        
        x1 = face_landmarks[start_idx].x
        y1 = face_landmarks[start_idx].y
        x2 = face_landmarks[end_idx].x
        y2 = face_landmarks[end_idx].y
        
        return x1, y1, x2, y2
    
    def get_eye_roi_px(self, face_landmarks, side, w, h):
        """
        Get eye ROI in pixel coords with NXP ROI_SCALE=2.
        Returns: (xmin, ymin, xmax, ymax) in pixel coordinates
        """
        x1_n, y1_n, x2_n, y2_n = self.get_eye_roi(face_landmarks, side)
        
        x1 = int(x1_n * w)
        y1 = int(y1_n * h)
        x2 = int(x2_n * w)
        y2 = int(y2_n * h)
        
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        half_w = abs(x2 - x1) // 2
        
        roi_xmin = max(0, mid_x - self.ROI_SCALE * half_w)
        roi_xmax = min(w, mid_x + self.ROI_SCALE * half_w)
        roi_ymin = max(0, mid_y - self.ROI_SCALE * half_w)
        roi_ymax = min(h, mid_y + self.ROI_SCALE * half_w)
        
        return roi_xmin, roi_ymin, roi_xmax, roi_ymax
    
    def predict(self, eye_crop_bgr, side=0):
        """
        Run iris/eye landmark inference on a cropped eye region.
        
        Args:
            eye_crop_bgr: BGR eye crop (any size, will be resized)
            side: 0 = left eye, 1 = right eye (right eye is flipped per NXP)
        
        Returns:
            iris_center: (x, y) normalized 0-1 relative to eye crop
            eye_contour: numpy array [N, 3] normalized eye contour points
            iris_landmarks: numpy array [5, 3] normalized iris points
        """
        if eye_crop_bgr is None or eye_crop_bgr.size == 0:
            return None, None, None
        
        # Right eye: flip input horizontally (NXP convention)
        if side == 1:
            eye_crop_bgr = cv2.flip(eye_crop_bgr, 1)
        
        # Preprocess: resize, BGR->RGB, normalize to [-1, 1] (NXP convention)
        img = cv2.resize(eye_crop_bgr, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 128.0) / 128.0
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details['index'], img)
        self.interpreter.invoke()
        
        # Parse outputs — NXP convention: eye contour at [0], iris at [1]
        eye_points = self.interpreter.get_tensor(self.eye_index).reshape(-1, 3).astype(np.float32)
        iris_points = self.interpreter.get_tensor(self.iris_index).reshape(-1, 3).astype(np.float32)
        
        # Normalize by (width, height, width) — NXP convention for 3D points
        norm_scale = np.array([self.input_w, self.input_h, self.input_w], dtype=np.float32)
        eye_points = eye_points / norm_scale
        iris_points = iris_points / norm_scale
        
        # Right eye: invert x-coordinates to undo the horizontal flip
        if side == 1:
            eye_points[:, 0] = 1.0 - eye_points[:, 0]
            iris_points[:, 0] = 1.0 - iris_points[:, 0]
        
        # Iris center = mean of iris points
        iris_center = (float(np.mean(iris_points[:, 0])), float(np.mean(iris_points[:, 1])))
        
        return iris_center, eye_points, iris_points
    
    def blinking_ratio(self, eye_contour, side=0):
        """
        Calculate eye openness ratio from eye contour landmarks (NXP method).
        Uses eye_height / eye_width from specific contour points.
        
        Args:
            eye_contour: [N, 3] normalized eye contour points
            side: 0 = left eye, 1 = right eye
        
        Returns:
            ratio: float, higher = more open
        """
        if eye_contour is None or len(eye_contour) < 15:
            return 0.0
        
        # NXP convention: points 0,8 for horizontal span; 4,12 for vertical span
        if side == 0:
            point_left = eye_contour[0]
            point_right = eye_contour[8]
        else:
            point_left = eye_contour[8]
            point_right = eye_contour[0]
        
        point_top = eye_contour[12]
        point_bottom = eye_contour[4]
        
        eye_width = np.sqrt((point_right[0] - point_left[0])**2 + (point_right[1] - point_left[1])**2)
        eye_height = np.sqrt((point_bottom[0] - point_top[0])**2 + (point_bottom[1] - point_top[1])**2)
        
        return eye_height / eye_width if eye_width > 0 else 0.0


# ============================================================
# BlazeFace Short-Range Face Detector (NPU)
# Replaces Haar Cascade with NXP face_detection_ptq_vela.tflite
# ============================================================

# Score limit to prevent IEEE 754 float overflow in sigmoid (NXP convention)
RAW_SCORE_LIMIT = 80
MIN_SUPPRESSION_THRESHOLD = 0.5


class BlazeFaceDetector:
    """
    BlazeFace short-range face detector on NPU.
    Input:  [1, 128, 128, 3] float32 normalized to [-1, 1]
    Output0: scores [1, 896, 1]
    Output1: boxes [1, 896, 16]
    
    Preprocessing: (pixel - 128) / 128.0 -> range [-1, 1]
    Box format per anchor: 16 values = 8 pairs of (x, y) coordinates
    (Matches NXP reference implementation exactly)
    """
    
    def __init__(self, model_path="face_detection_ptq_vela.tflite", threshold=0.65):
        self.interpreter = load_npu_model(model_path)
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details['shape']  # [1, 128, 128, 3]
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        
        # NXP convention: scores at output[0], boxes at output[1]
        self.score_index = self.output_details[0]['index']
        self.bbox_index = self.output_details[1]['index']
        
        # SSD anchor configuration (matches NXP face_detection_short_range_common.pbtxt)
        self.ssd_opts = {
            'num_layers': 4,
            'input_size_height': self.input_h,
            'input_size_width': self.input_w,
            'anchor_offset_x': 0.5,
            'anchor_offset_y': 0.5,
            'strides': [8, 16, 16, 16],
            'interpolated_scale_aspect_ratio': 1.0,
        }
        
        self.anchors = self._ssd_generate_anchors(self.ssd_opts)
        self.threshold = threshold
        self.last_box = None
        self.no_face_count = 0
        
        print(f"[BlazeFace] Input: {self.input_shape}, anchors: {len(self.anchors)}")
        for i, od in enumerate(self.output_details):
            print(f"[BlazeFace] Output {i}: {od['shape']}, name={od['name']}")
    
    def _ssd_generate_anchors(self, opts):
        """
        Generate SSD anchors — exact NXP/MediaPipe implementation.
        Groups same-stride layers and generates 2 anchors per grouped layer
        when interpolated_scale_aspect_ratio == 1.0
        (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
        """
        layer_id = 0
        num_layers = opts['num_layers']
        strides = opts['strides']
        input_height = opts['input_size_height']
        input_width = opts['input_size_width']
        anchor_offset_x = opts['anchor_offset_x']
        anchor_offset_y = opts['anchor_offset_y']
        interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
        
        anchors = []
        while layer_id < num_layers:
            last_same_stride_layer = layer_id
            repeats = 0
            # Group consecutive layers with same stride together
            while (last_same_stride_layer < num_layers
                   and strides[last_same_stride_layer] == strides[layer_id]):
                last_same_stride_layer += 1
                # 2 anchors per layer when interpolated_scale_aspect_ratio == 1.0
                repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
            
            stride = strides[layer_id]
            feature_map_height = input_height // stride
            feature_map_width = input_width // stride
            
            for y in range(feature_map_height):
                y_center = (y + anchor_offset_y) / feature_map_height
                for x in range(feature_map_width):
                    x_center = (x + anchor_offset_x) / feature_map_width
                    for _ in range(repeats):
                        anchors.append((x_center, y_center))
            
            layer_id = last_same_stride_layer
        
        return np.array(anchors, dtype=np.float32)
    
    def _decode_boxes(self, raw_boxes):
        """
        Decode raw box predictions — exact NXP/MediaPipe implementation.
        (reference: mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc)
        """
        # width == height so scale is the same across the board
        scale = self.input_shape[1]
        num_points = raw_boxes.shape[-1] // 2
        # Scale all values (applies to positions, width, and height alike)
        boxes = raw_boxes.reshape(-1, num_points, 2) / scale
        # Adjust center coordinates and key points to anchor positions
        boxes[:, 0] += self.anchors
        for i in range(2, num_points):
            boxes[:, i] += self.anchors
        # Convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
        center = np.array(boxes[:, 0])
        half_size = boxes[:, 1] / 2
        boxes[:, 0] = center - half_size
        boxes[:, 1] = center + half_size
        # Only need boxes xmin, ymin, xmax, ymax
        boxes = boxes[:, 0:2, :].reshape(-1, 4)
        return boxes
    
    def _get_sigmoid_scores(self, raw_scores):
        """Apply sigmoid with clamping to prevent overflow (NXP convention)"""
        raw_scores = np.clip(raw_scores, -RAW_SCORE_LIMIT, RAW_SCORE_LIMIT)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    
    def _overlap_similarity(self, box1, box2):
        """IoU similarity between two bounding boxes"""
        if box1 is None or box2 is None:
            return 0
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        x3_min = max(x1_min, x2_min)
        x3_max = min(x1_max, x2_max)
        y3_min = max(y1_min, y2_min)
        y3_max = min(y1_max, y2_max)
        intersect_area = max(0, x3_max - x3_min) * max(0, y3_max - y3_min)
        denominator = box1_area + box2_area - intersect_area
        return intersect_area / denominator if denominator > 0.0 else 0.0
    
    def _non_maximum_suppression(self, boxes, scores):
        """Non-maximum suppression — exact NXP implementation"""
        candidates_list = []
        for i in range(np.size(boxes, 0)):
            candidates_list.append((boxes[i], scores[i]))
        candidates_list = sorted(candidates_list, key=lambda x: x[1], reverse=True)
        kept_list = []
        for sorted_box, sorted_score in candidates_list:
            suppressed = False
            for kept in kept_list:
                similarity = self._overlap_similarity(kept, sorted_box)
                if similarity > MIN_SUPPRESSION_THRESHOLD:
                    suppressed = True
                    break
            if not suppressed:
                kept_list.append(sorted_box)
        return kept_list
    
    def detect(self, frame, conf_threshold=None):
        """
        Detect face in frame.
        Returns: (x1, y1, x2, y2) in pixel coordinates, or None
        """
        if conf_threshold is None:
            conf_threshold = self.threshold
        
        h, w = frame.shape[:2]
        
        # Preprocess: resize, BGR->RGB, normalize to [-1, 1] (NXP convention)
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 128.0) / 128.0
        img = np.expand_dims(img, axis=0)  # [1, 128, 128, 3]
        
        # Run inference
        self.interpreter.set_tensor(self.input_details['index'], img)
        self.interpreter.invoke()
        
        # Get outputs — NXP convention: scores at [0], boxes at [1]
        raw_scores = self.interpreter.get_tensor(self.score_index)
        raw_boxes = self.interpreter.get_tensor(self.bbox_index)
        
        # Decode boxes and scores (NXP exact algorithm)
        boxes = self._decode_boxes(raw_boxes)
        scores = self._get_sigmoid_scores(raw_scores)
        
        # Filter by threshold
        score_above_threshold = scores > conf_threshold
        filtered_indices = np.argwhere(score_above_threshold)
        if len(filtered_indices) == 0:
            self.no_face_count += 1
            if self.no_face_count > 10:
                self.last_box = None
            return self.last_box
        
        filtered_boxes = boxes[filtered_indices[:, 1], :]
        filtered_scores = scores[score_above_threshold]
        
        # Non-maximum suppression (NXP exact algorithm)
        nms_result = self._non_maximum_suppression(filtered_boxes, filtered_scores)
        
        if len(nms_result) == 0:
            self.no_face_count += 1
            if self.no_face_count > 10:
                self.last_box = None
            return self.last_box
        
        # Take the first (highest confidence) detection
        best = nms_result[0]
        
        # Scale from normalized [0, 1] to pixel coordinates
        bx1 = int(np.clip(best[0], 0, 1) * w)
        by1 = int(np.clip(best[1], 0, 1) * h)
        bx2 = int(np.clip(best[2], 0, 1) * w)
        by2 = int(np.clip(best[3], 0, 1) * h)
        
        # Validate box size
        if (bx2 - bx1) < 20 or (by2 - by1) < 20:
            self.no_face_count += 1
            if self.no_face_count > 10:
                self.last_box = None
            return self.last_box
        
        self.last_box = (bx1, by1, bx2, by2)
        self.no_face_count = 0
        return self.last_box


# ============================================================
# Palm Detector (NPU) — replaces MediaPipe Hands detection stage
# Uses same SSD anchor pattern as BlazeFace but 192×192 input
# ============================================================

class PalmDetector:
    """
    Palm detector using palm_detection_full_quant_vela.tflite
    Input:  [1, 192, 192, 3] float32 normalized to [-1, 1]
    Output0: scores [1, 2016, 1]
    Output1: boxes [1, 2016, 18]  (center_x, center_y, w, h + 7 keypoints × 2)
    """
    
    def __init__(self, model_path="palm_detection_full_quant_vela.tflite", threshold=0.6):
        self.interpreter = load_npu_model(model_path)
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details['shape']  # [1, 192, 192, 3]
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        
        # Auto-detect score vs box outputs by shape (not position)
        # Scores: last dim = 1, Boxes: last dim = 18 (cx, cy, w, h + 7 keypoints × 2)
        self.score_index = None
        self.bbox_index = None
        self.score_detail = None
        self.bbox_detail = None
        for i, od in enumerate(self.output_details):
            last_dim = od['shape'][-1]
            if last_dim == 1:
                self.score_index = od['index']
                self.score_detail = od
                print(f"[PalmDet] Output {i} → SCORES (shape={od['shape']}, dtype={od['dtype']})")
            elif last_dim == 18:
                self.bbox_index = od['index']
                self.bbox_detail = od
                print(f"[PalmDet] Output {i} → BOXES (shape={od['shape']}, dtype={od['dtype']})")
            else:
                print(f"[PalmDet] Output {i} → UNKNOWN (shape={od['shape']})")
        
        # Fallback if shapes don't match expected pattern
        if self.score_index is None or self.bbox_index is None:
            print(f"[PalmDet] WARNING: Could not auto-detect outputs, using positional fallback")
            self.score_index = self.output_details[0]['index']
            self.bbox_index = self.output_details[1]['index']
            self.score_detail = self.output_details[0]
            self.bbox_detail = self.output_details[1]
        
        # SSD anchors — same pattern as BlazeFace, different input size
        # 192/8=24 → 24×24×2=1152, 192/16=12 → 12×12×6=864, total=2016
        self.ssd_opts = {
            'num_layers': 4,
            'input_size_height': self.input_h,
            'input_size_width': self.input_w,
            'anchor_offset_x': 0.5,
            'anchor_offset_y': 0.5,
            'strides': [8, 16, 16, 16],
            'interpolated_scale_aspect_ratio': 1.0,
        }
        
        self.anchors = self._ssd_generate_anchors(self.ssd_opts)
        self.threshold = threshold
        
        print(f"[PalmDet] Input: {self.input_shape}, anchors: {len(self.anchors)}")
        for i, od in enumerate(self.output_details):
            print(f"[PalmDet] Output {i}: {od['shape']}, name={od['name']}")
    
    def _ssd_generate_anchors(self, opts):
        """Generate SSD anchors — same algorithm as BlazeFace"""
        layer_id = 0
        num_layers = opts['num_layers']
        strides = opts['strides']
        input_height = opts['input_size_height']
        input_width = opts['input_size_width']
        anchor_offset_x = opts['anchor_offset_x']
        anchor_offset_y = opts['anchor_offset_y']
        interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
        
        anchors = []
        while layer_id < num_layers:
            last_same_stride_layer = layer_id
            repeats = 0
            while (last_same_stride_layer < num_layers
                   and strides[last_same_stride_layer] == strides[layer_id]):
                last_same_stride_layer += 1
                repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
            
            stride = strides[layer_id]
            feature_map_height = input_height // stride
            feature_map_width = input_width // stride
            
            for y in range(feature_map_height):
                y_center = (y + anchor_offset_y) / feature_map_height
                for x in range(feature_map_width):
                    x_center = (x + anchor_offset_x) / feature_map_width
                    for _ in range(repeats):
                        anchors.append((x_center, y_center))
            
            layer_id = last_same_stride_layer
        
        return np.array(anchors, dtype=np.float32)
    
    def _decode_boxes(self, raw_boxes):
        """Decode palm detection boxes — same as BlazeFace"""
        scale = self.input_shape[1]
        num_points = raw_boxes.shape[-1] // 2
        boxes = raw_boxes.reshape(-1, num_points, 2) / scale
        boxes[:, 0] += self.anchors
        for i in range(2, num_points):
            boxes[:, i] += self.anchors
        center = np.array(boxes[:, 0])
        half_size = boxes[:, 1] / 2
        boxes[:, 0] = center - half_size
        boxes[:, 1] = center + half_size
        boxes = boxes[:, 0:2, :].reshape(-1, 4)
        return boxes
    
    def detect(self, frame, conf_threshold=None, max_hands=2):
        """
        Detect palms in frame.
        Returns: list of (x1, y1, x2, y2) in pixel coords, up to max_hands
        """
        if conf_threshold is None:
            conf_threshold = self.threshold
        
        h, w = frame.shape[:2]
        
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 128.0) / 128.0
        img = np.expand_dims(img, axis=0)
        
        self.interpreter.set_tensor(self.input_details['index'], img)
        self.interpreter.invoke()
        
        # Dequantize outputs — critical for INT8 quantized models
        raw_scores = dequantize_tensor(self.interpreter, self.score_index, self.score_detail)
        raw_boxes = dequantize_tensor(self.interpreter, self.bbox_index, self.bbox_detail)
        
        # Debug: print raw score stats periodically
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0
        if self._debug_count % 30 == 0:
            sr = raw_scores.flatten()
            print(f"[PalmDet] score_shape={raw_scores.shape}, box_shape={raw_boxes.shape}")
            print(f"[PalmDet] raw_scores: min={sr.min():.3f}, max={sr.max():.3f}, mean={sr.mean():.3f}")
            sig_max = 1.0 / (1.0 + np.exp(-np.clip(sr.max(), -20, 20)))
            print(f"[PalmDet] sigmoid(max_score)={sig_max:.4f}, threshold={conf_threshold}")
            # Show dequant info once
            if self._debug_count == 0:
                raw_int = self.interpreter.get_tensor(self.score_index)
                print(f"[PalmDet] score tensor dtype={raw_int.dtype}, "
                      f"quant_params={self.score_detail.get('quantization_parameters', 'N/A')}")
                raw_int_b = self.interpreter.get_tensor(self.bbox_index)
                print(f"[PalmDet] box tensor dtype={raw_int_b.dtype}, "
                      f"quant_params={self.bbox_detail.get('quantization_parameters', 'N/A')}")
        
        boxes = self._decode_boxes(raw_boxes)
        scores_raw = raw_scores.flatten()
        scores = np.clip(scores_raw, -RAW_SCORE_LIMIT, RAW_SCORE_LIMIT)
        scores = 1.0 / (1.0 + np.exp(-scores))
        
        # Filter by threshold
        mask = scores > conf_threshold
        if not np.any(mask):
            return []
        
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        
        # NMS
        candidates = sorted(zip(filtered_boxes, filtered_scores), key=lambda x: x[1], reverse=True)
        kept = []
        for box, score in candidates:
            suppressed = False
            for k in kept:
                # IoU check
                x1i = max(box[0], k[0]); y1i = max(box[1], k[1])
                x2i = min(box[2], k[2]); y2i = min(box[3], k[3])
                inter = max(0, x2i - x1i) * max(0, y2i - y1i)
                a1 = (box[2]-box[0])*(box[3]-box[1])
                a2 = (k[2]-k[0])*(k[3]-k[1])
                iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0
                if iou > MIN_SUPPRESSION_THRESHOLD:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(box)
                if len(kept) >= max_hands:
                    break
        
        # Convert to pixel coords with margin for hand landmark crop
        results = []
        for box in kept:
            bx1 = int(np.clip(box[0], 0, 1) * w)
            by1 = int(np.clip(box[1], 0, 1) * h)
            bx2 = int(np.clip(box[2], 0, 1) * w)
            by2 = int(np.clip(box[3], 0, 1) * h)
            # Add 20% margin for hand landmark model
            bw = bx2 - bx1
            bh = by2 - by1
            mx = int(bw * 0.2)
            my = int(bh * 0.2)
            bx1 = max(0, bx1 - mx)
            by1 = max(0, by1 - my)
            bx2 = min(w, bx2 + mx)
            by2 = min(h, by2 + my)
            if (bx2 - bx1) > 15 and (by2 - by1) > 15:
                results.append((bx1, by1, bx2, by2))
        
        return results


# ============================================================
# Hand Landmark Detector (NPU) — replaces MediaPipe Hands landmark stage
# ============================================================

class HandLandmarkDetector:
    """
    Hand Landmark detector using hand_landmark_full_quant_vela.tflite
    Input:  [1, 224, 224, 3] float32 normalized to [-1, 1]
    Output0: [1, 1]  — handedness score (>0.5 = right hand)
    Output1: [1, 63] — 21 landmarks × 3 (x, y, z)
    Output2: [1, 1]  — hand presence/confidence
    Output3: [1, 63] — world landmarks (3D in meters)
    """
    
    # 21 MediaPipe hand landmark names for reference
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    def __init__(self, model_path="hand_landmark_full_quant_vela.tflite"):
        self.interpreter = load_npu_model(model_path)
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details['shape']  # [1, 224, 224, 3]
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        
        # Map outputs by shape: [1,1] = scores, [1,63] = landmarks
        self.handedness_idx = None
        self.landmark_idx = None
        self.presence_idx = None
        self.world_lm_idx = None
        # Store full output details for dequantization
        self.handedness_detail = None
        self.landmark_detail = None
        self.presence_detail = None
        self.world_lm_detail = None
        
        score_indices = []
        lm_indices = []
        for i, od in enumerate(self.output_details):
            shape = tuple(od['shape'])
            if shape == (1, 1):
                score_indices.append(i)
            elif shape == (1, 63):
                lm_indices.append(i)
        
        # Assign based on typical MediaPipe output order
        if len(score_indices) >= 2:
            self.handedness_idx = self.output_details[score_indices[0]]['index']
            self.handedness_detail = self.output_details[score_indices[0]]
            self.presence_idx = self.output_details[score_indices[1]]['index']
            self.presence_detail = self.output_details[score_indices[1]]
        if len(lm_indices) >= 2:
            self.landmark_idx = self.output_details[lm_indices[0]]['index']
            self.landmark_detail = self.output_details[lm_indices[0]]
            self.world_lm_idx = self.output_details[lm_indices[1]]['index']
            self.world_lm_detail = self.output_details[lm_indices[1]]
        
        print(f"[HandLandmark] Input: {self.input_shape}, dtype={self.input_details['dtype']}")
        for i, od in enumerate(self.output_details):
            print(f"[HandLandmark] Output {i}: {od['shape']}, dtype={od['dtype']}, name={od['name']}")
    
    def predict(self, hand_crop_bgr):
        """
        Run hand landmark inference on a cropped palm region.
        
        Args:
            hand_crop_bgr: BGR hand crop (any size, will be resized to 224×224)
        
        Returns:
            presence: float (0-1) hand presence confidence
            landmarks: list of 21 LandmarkPoint objects (normalized 0-1 relative to crop)
            handedness: float (>0.5 = right hand)
        """
        if hand_crop_bgr is None or hand_crop_bgr.size == 0:
            return 0.0, None, 0.0
        
        # Preprocess: resize, BGR->RGB, normalize to [-1, 1]
        img = cv2.resize(hand_crop_bgr, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 128.0) / 128.0
        img = np.expand_dims(img, axis=0)
        
        self.interpreter.set_tensor(self.input_details['index'], img)
        self.interpreter.invoke()
        
        # Parse outputs — dequantize INT8 if needed
        presence = 0.0
        handedness = 0.0
        if self.presence_idx is not None:
            raw_presence = float(dequantize_tensor(
                self.interpreter, self.presence_idx, self.presence_detail).flatten()[0])
            presence = 1.0 / (1.0 + np.exp(-raw_presence))  # sigmoid
        if self.handedness_idx is not None:
            raw_hand = float(dequantize_tensor(
                self.interpreter, self.handedness_idx, self.handedness_detail).flatten()[0])
            handedness = 1.0 / (1.0 + np.exp(-raw_hand))  # sigmoid
        
        if self.landmark_idx is None:
            return presence, None, handedness
        
        raw_lm = dequantize_tensor(
            self.interpreter, self.landmark_idx, self.landmark_detail).flatten().astype(np.float32)
        if raw_lm.size < 63:
            return presence, None, handedness
        
        # Reshape to [21, 3], normalize to 0-1
        lm = raw_lm[:63].reshape(21, 3)
        lm[:, 0] /= self.input_w
        lm[:, 1] /= self.input_h
        lm[:, 2] /= self.input_w
        
        # Create LandmarkPoint list
        hand_landmarks = [LandmarkPoint(lm[i, 0], lm[i, 1], lm[i, 2]) for i in range(21)]
        
        return presence, hand_landmarks, handedness


# ============================================================
# Landmark Adapter: converts raw landmarks to pixel coordinates
# Mimics MediaPipe landmark interface for compatibility
# ============================================================

class LandmarkPoint:
    """Mimics mediapipe landmark with x, y, z attributes (normalized 0-1)"""
    __slots__ = ['x', 'y', 'z']
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def crop_face_with_margin(frame, box, margin=0.3):
    """Crop face from frame with margin, return crop and coordinates"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    fw, fh = x2 - x1, y2 - y1
    
    # Add margin
    mx = int(fw * margin)
    my = int(fh * margin)
    
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)
    
    crop = frame[cy1:cy2, cx1:cx2]
    return crop, (cx1, cy1, cx2, cy2)


def get_eye_crop(frame, landmarks, eye_indices, w, h, margin=0.4):
    """Extract eye region crop from frame using landmark indices"""
    xs = [landmarks[i].x * w for i in eye_indices]
    ys = [landmarks[i].y * h for i in eye_indices]
    
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    
    ew = x_max - x_min
    eh = y_max - y_min
    mx = int(ew * margin)
    my = int(eh * margin) + 5  # Extra vertical margin
    
    x1 = max(0, x_min - mx)
    y1 = max(0, y_min - my)
    x2 = min(frame.shape[1], x_max + mx)
    y2 = min(frame.shape[0], y_max + my)
    
    if x2 <= x1 or y2 <= y1:
        return None, None
    
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--ear_threshold', type=float, default=0.165)
parser.add_argument('--eye_closed_frames_threshold', type=int, default=9)
parser.add_argument('--blink_rate_threshold', type=int, default=5)
parser.add_argument('--mar_threshold', type=float, default=0.65)
parser.add_argument('--yawn_threshold', type=int, default=3)
parser.add_argument('--frame_width', type=int, default=1280)
parser.add_argument('--frame_height', type=int, default=720)
parser.add_argument('--gaze_deviation_threshold', type=float, default=0.025)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--head_turn_threshold', type=float, default=0.22, help='0.22 for 45deg side cam, 0.08 for front cam')
parser.add_argument('--hand_near_face_px', type=int, default=200)
parser.add_argument('--calibration_time', type=int, default=10, help='10s calibration (median-based, matches ref)')
parser.add_argument('--no_face_display', action='store_true')
parser.add_argument('--no_mesh_display', action='store_true')
parser.add_argument('--mjpeg_port', type=int, default=8080)
parser.add_argument('--enable_streaming', action='store_true', default=True)
parser.add_argument('--headless', action='store_true', default=True)
parser.add_argument('--camera_device', type=str, default='auto')
parser.add_argument('--crop_center', type=float, default=0.0)
parser.add_argument('--face_detection_model', type=str, default='face_detection_ptq_vela.tflite')
parser.add_argument('--face_landmark_model', type=str, default='face_landmark_ptq_vela.tflite')
parser.add_argument('--iris_landmark_model', type=str, default='iris_landmark_ptq_vela.tflite')
parser.add_argument('--face_conf_threshold', type=float, default=0.5, help="Face confidence threshold from landmark model")
parser.add_argument('--face_det_threshold', type=float, default=0.65, help="Face detection confidence threshold")
parser.add_argument('--palm_detection_model', type=str, default='palm_detection_full_quant_vela.tflite')
parser.add_argument('--hand_landmark_model', type=str, default='hand_landmark_full_quant_vela.tflite')
parser.add_argument('--palm_det_threshold', type=float, default=0.50, help="Palm detection confidence threshold (MediaPipe default=0.5)")
parser.add_argument('--hand_presence_threshold', type=float, default=0.6, help="Hand landmark presence threshold (0.5=noise, 0.6+=real hand)")

args = parser.parse_args()

# ============================================================
# Initialize NPU models (replaces MediaPipe)
# ============================================================

print("[DMS] Loading NPU models...")
face_detector = BlazeFaceDetector(args.face_detection_model)
face_landmark_detector = FaceLandmarkDetector(args.face_landmark_model)
iris_landmark_detector = IrisLandmarkDetector(args.iris_landmark_model)
palm_detector = PalmDetector(args.palm_detection_model, threshold=args.palm_det_threshold)
hand_landmark_detector = HandLandmarkDetector(args.hand_landmark_model)
print("[DMS] All 5 models loaded on NPU (face_det + face_lm + iris + palm_det + hand_lm)")

# MediaPipe landmark indices (same as original dmsv8.py)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
# NOTE: MediaPipe 478-point model has iris at 469-477, but NXP model only outputs 468 points.
# Use eye corner indices as fallback for iris center estimation.
LEFT_IRIS_FALLBACK = [33, 133]    # left eye corners (valid in 468-point model)
RIGHT_IRIS_FALLBACK = [362, 263]  # right eye corners (valid in 468-point model)
NOSE_TIP = 1
LEFT_EAR_TIP = 234
RIGHT_EAR_TIP = 454

# Auto-detect available camera
def find_available_camera():
    print("[Camera] Auto-detecting available cameras...")
    video_devices = sorted(glob.glob('/dev/video*'))
    if video_devices:
        print(f"[Camera] Found devices: {', '.join(video_devices)}")
        for device in video_devices:
            try:
                device_num = int(device.split('video')[-1])
                test_cap = cv2.VideoCapture(device_num)
                if test_cap.isOpened():
                    test_cap.release()
                    print(f"[Camera] Selected: {device} (device {device_num})")
                    return device_num
                test_cap.release()
            except:
                continue
    
    print("[Camera] /dev/video* not found, trying numeric indices...")
    for i in range(6):
        try:
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                test_cap.release()
                print(f"[Camera] Selected: camera {i}")
                return i
            test_cap.release()
        except:
            continue
    
    print("[Camera] ERROR: No cameras found!")
    return None

print(f"[Camera] Requested device: {args.camera_device}")

try:
    if args.camera_device == 'auto':
        device_num = find_available_camera()
        if device_num is None:
            print("[ERROR] No camera available")
            exit(1)
    else:
        if '/dev/video' in args.camera_device:
            device_num = int(args.camera_device.split('video')[-1])
        else:
            device_num = int(args.camera_device)
    
    print(f"[Camera] Opening camera {device_num}...")
    cap = cv2.VideoCapture(device_num)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {device_num}")
        exit(1)
    
    print("[Camera] Opened successfully")
except Exception as e:
    print(f"[ERROR] Camera initialization failed: {e}")
    exit(1)

# Start MJPEG streaming
mjpeg_server = None
if args.enable_streaming:
    mjpeg_server = MJPEGServer(host='0.0.0.0', port=args.mjpeg_port)
    if mjpeg_server.start():
        print(f"[DMS] Stream ready at port {args.mjpeg_port}")
    else:
        mjpeg_server = None

eye_closure_counter = 0
blink_counter = 0
blink_total = 0
blink_timer = time.time()
yawn_counter = 0
mar_deque = deque(maxlen=30)

ALERT_DURATION = 3
active_alerts = defaultdict(lambda: [None, 0])
calibration_mode = True
calibration_start_time = time.time()
calibration_duration = args.calibration_time
gaze_center = 0.5
head_center_x = 0.5
head_center_y = 0.5

# Collect multiple samples during calibration for robust baseline (matches MediaPipe ref)
calib_gaze_x_samples = deque(maxlen=600)
calib_head_x_samples = deque(maxlen=600)
calib_head_y_samples = deque(maxlen=600)
calib_min_samples = 15

# EAR smoothing (adaptive: fast close, slow open — matches MediaPipe ref)
smoothed_ear = None
prev_ear = None
prev_eye_state = None
eye_close_start_time = None
prev_frame_time = time.time()

# Gaze confirmation streaks (require N consecutive frames — matches MediaPipe ref)
GAZE_CONFIRM_FRAMES = 3
gaze_left_confirm_streak = 0
gaze_right_confirm_streak = 0
gaze_alerted_left = False
gaze_alerted_right = False
GAZE_ALERT_COOLDOWN = 3.0
last_gaze_left_alert_time = 0
last_gaze_right_alert_time = 0

# Head position smoothing (EMA alpha=0.3 — matches MediaPipe ref)
sm_head_x_signed = 0.0
sm_head_y_signed = 0.0

print(f"[Calibration] Starting {args.calibration_time}s calibration period...")
print(f"[Calibration] Look straight ahead and keep face visible")


def add_alert(frame, message):
    ts = datetime.now().strftime("%H:%M:%S")
    active_alerts[f"{ts} {message}"] = time.time()
    print(f"[ALERT {ts}] {message}")
    return message


def get_aspect_ratio(landmarks, eye_indices, w, h):
    def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])

    A = np.linalg.norm(pt(eye_indices[1]) - pt(eye_indices[5]))
    B = np.linalg.norm(pt(eye_indices[2]) - pt(eye_indices[4]))
    C = np.linalg.norm(pt(eye_indices[0]) - pt(eye_indices[3]))

    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear


def get_mar(landmarks, mouth_idx, w, h):
    top = np.array([landmarks[mouth_idx[0]].x * w, landmarks[mouth_idx[0]].y * h])
    bottom = np.array([landmarks[mouth_idx[1]].x * w, landmarks[mouth_idx[1]].y * h])
    left = np.array([landmarks[mouth_idx[2]].x * w, landmarks[mouth_idx[2]].y * h])
    right = np.array([landmarks[mouth_idx[3]].x * w, landmarks[mouth_idx[3]].y * h])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal if horizontal > 0 else 0


def get_iris_center_from_landmarks(landmarks, indices, w, h):
    """Get iris center from face landmark indices (without iris model)"""
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    return np.mean(points, axis=0)


def hand_near_ear(face_landmarks, hand_lm_list, w, h):
    """
    Check if hand fingertips are near ear landmarks.
    Matches ref: asymmetric thresholds for right-side 45° camera.
    Uses face-height-relative radii + nose Y filtering.
    """
    try:
        xs = [lm.x * w for lm in face_landmarks]
        ys = [lm.y * h for lm in face_landmarks]
        face_h = max(1.0, float(max(ys) - min(ys)))
    except Exception:
        face_h = float(h)

    # Nose landmark as divider (landmark index 1 in 468-point mesh)
    try:
        nose_y = face_landmarks[1].y * h
    except Exception:
        nose_y = h * 0.5

    ear_l = np.array([face_landmarks[LEFT_EAR_TIP].x * w, face_landmarks[LEFT_EAR_TIP].y * h])
    ear_r = np.array([face_landmarks[RIGHT_EAR_TIP].x * w, face_landmarks[RIGHT_EAR_TIP].y * h])
    tips = [4, 8, 12, 16, 20]  # fingertip landmark indices in hand model

    # Asymmetric radius for right-side 45° camera (matches ref)
    r_pix_right = 0.20 * face_h  # right ear (closer to camera)
    r_pix_left  = 0.60 * face_h  # left ear (farther, perspective distortion)

    # Nose filter boundary — only consider fingertips above nose level
    nose_filter_right = nose_y + (0.12 * face_h)
    nose_filter_left  = nose_y + (0.15 * face_h)

    near_l = 0
    near_r = 0
    for idx in tips:
        if idx >= len(hand_lm_list):
            continue
        lm = hand_lm_list[idx]
        hx, hy = lm.x * w, lm.y * h

        # Right ear check (with nose filtering)
        if hy <= nose_filter_right:
            if np.hypot(hx - ear_r[0], hy - ear_r[1]) <= r_pix_right:
                near_r += 1

        # Left ear check (with nose filtering)
        if hy <= nose_filter_left:
            if np.hypot(hx - ear_l[0], hy - ear_l[1]) <= r_pix_left:
                near_l += 1

    return (near_l >= 1) or (near_r >= 1)


def hand_near_face(face_center, hand_lm_list, w, h, threshold_px=200):
    """Check if any hand landmark is near face center (matches dmsv8 exactly)"""
    fcx, fcy = face_center
    for lm in hand_lm_list:
        x, y = int(lm.x * w), int(lm.y * h)
        if np.hypot(fcx - x, fcy - y) < threshold_px:
            return True
    return False


eye_closed = 0
head_turn = 0
hands_free = False
head_tilt = 0
head_droop = 0
yawn_flag = False
msg = ""
last_msg = "Normal and Active Driving"
fid = 0
fps = 0.0
fps_start_time = time.time()
fps_frame_count = 0

# Iris frame-skipping: run iris model every N frames, reuse result otherwise
IRIS_SKIP_FRAMES = 2  # run iris every 2nd frame (saves ~8ms avg per frame)
cached_iris_left_center = None
cached_iris_right_center = None
cached_left_eye_contour = None
cached_right_eye_contour = None

# --- Anti-flicker: temporal smoothing (matches MediaPipe internal filtering) ---
# Face box EMA smoothing (stabilizes crop → stabilizes landmarks)
smoothed_box = None
BOX_SMOOTH_ALPHA = 0.4  # 0.4 = responsive yet stable (lower = smoother but laggier)
# Landmark EMA smoothing (MediaPipe does this internally via its FilterStabilizer)
smoothed_landmarks = None  # numpy [468, 3]
LM_SMOOTH_ALPHA = 0.5     # 0.5 = balanced (MediaPipe uses ~0.3-0.5 equivalent)

# ---- MediaPipe-style hand landmark drawing ----
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (5,9),(9,10),(10,11),(11,12),      # middle
    (9,13),(13,14),(14,15),(15,16),    # ring
    (13,17),(17,18),(18,19),(19,20),   # pinky
    (0,17),                            # palm base
]
# Finger color groups (BGR): matches MediaPipe visualization
HAND_FINGER_COLORS = {
    'thumb':  (48, 48, 255),    # red
    'index':  (48, 255, 48),    # green
    'middle': (255, 187, 0),    # blue-ish
    'ring':   (0, 204, 255),    # orange
    'pinky':  (255, 0, 204),    # pink
    'palm':   (200, 200, 200),  # gray
}
# Map each landmark index to its finger group
LM_TO_FINGER = {}
for i in range(5):  LM_TO_FINGER[i] = 'thumb'
for i in range(5,9):  LM_TO_FINGER[i] = 'index'
for i in range(9,13):  LM_TO_FINGER[i] = 'middle'
for i in range(13,17): LM_TO_FINGER[i] = 'ring'
for i in range(17,21): LM_TO_FINGER[i] = 'pinky'
# Map each connection to a color
CONN_COLORS = []
for (a, b) in HAND_CONNECTIONS:
    if a == 0 and b in (5, 17):  # palm base connections
        CONN_COLORS.append(HAND_FINGER_COLORS['palm'])
    elif 5 == a and b == 9:
        CONN_COLORS.append(HAND_FINGER_COLORS['palm'])
    elif 9 == a and b == 13:
        CONN_COLORS.append(HAND_FINGER_COLORS['palm'])
    elif 13 == a and b == 17:
        CONN_COLORS.append(HAND_FINGER_COLORS['palm'])
    else:
        CONN_COLORS.append(HAND_FINGER_COLORS.get(LM_TO_FINGER.get(b, 'palm'), (200,200,200)))

def draw_hand_landmarks_mp(frame, hand_lm_list, w, h):
    """Draw hand landmarks MediaPipe-style: connections + colored joints."""
    pts = []
    for lm in hand_lm_list:
        pts.append((int(lm.x * w), int(lm.y * h)))
    # Draw connections first (behind joints)
    for idx, (a, b) in enumerate(HAND_CONNECTIONS):
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], CONN_COLORS[idx], 2, cv2.LINE_AA)
    # Draw joint circles on top
    for i, pt in enumerate(pts):
        finger = LM_TO_FINGER.get(i, 'palm')
        color = HAND_FINGER_COLORS[finger]
        radius = 5 if i in (0, 4, 8, 12, 16, 20) else 3  # larger for wrist + fingertips
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius, (255, 255, 255), 1, cv2.LINE_AA)  # white border

# Hand detection frame-skipping (palm det = 13.57ms, expensive)
PALM_SKIP_FRAMES = 3  # run palm detection every 3rd frame
cached_palm_boxes = []
cached_hand_landmarks_list = []  # list of (hand_lm_list, handedness) tuples
hand_lost_count = 0
MAX_HAND_LOST = 2  # clear cache after N palm-detection cycles without detection

# Hand alert confirmation (prevent random single-frame false positives)
HAND_CONFIRM_FRAMES = 5  # require N consecutive frames with hand detected before alerting
hand_near_ear_streak = 0
hand_near_face_streak = 0
texting_streak = 0

while cap.isOpened():
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to read frame")
        break

    fid += 1
    
    # Crop center portion for wide-angle cameras
    if args.crop_center > 0 and args.crop_center <= 0.5:
        h_orig, w_orig = frame.shape[:2]
        crop_h = int(h_orig * args.crop_center)
        crop_w = int(w_orig * args.crop_center)
        frame = frame[crop_h:h_orig-crop_h, crop_w:w_orig-crop_w]
        if fid == 1:
            print(f"[Camera] Cropped from {w_orig}x{h_orig} to {frame.shape[1]}x{frame.shape[0]}")
    
    if fid == 1:
        print(f"[Camera] Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"[Camera] Frame format: {frame.dtype}")
        print("[DMS] Processing started...")

    h, w = frame.shape[:2]
    current_time = time.time()
    
    if args.no_face_display:
        frame = np.zeros_like(frame)

    # ---- STEP 1: Detect face (bounding box) via BlazeFace NPU ----
    raw_face_box = face_detector.detect(frame, conf_threshold=args.face_det_threshold)
    
    # Smooth face box with EMA to prevent crop jitter → landmark flicker
    if raw_face_box is not None:
        if smoothed_box is None:
            smoothed_box = np.array(raw_face_box, dtype=np.float64)
        else:
            raw_arr = np.array(raw_face_box, dtype=np.float64)
            smoothed_box = BOX_SMOOTH_ALPHA * raw_arr + (1 - BOX_SMOOTH_ALPHA) * smoothed_box
        face_box = tuple(int(v) for v in smoothed_box)
    else:
        face_box = raw_face_box
        if raw_face_box is None:
            smoothed_box = None  # Reset when face lost
    
    # ---- STEP 2: Run face landmark on detected face ----
    landmarks = None
    face_confidence = 0.0
    
    if face_box is not None:
        face_crop, crop_coords = crop_face_with_margin(frame, face_box, margin=0.3)
        
        if face_crop is not None and face_crop.size > 0:
            face_confidence, raw_lms = face_landmark_detector.predict(face_crop)
            
            if raw_lms is not None and face_confidence > args.face_conf_threshold:
                # Convert landmarks from crop-relative (0-1) to frame-relative (0-1)
                cx1, cy1, cx2, cy2 = crop_coords
                crop_w_px = cx2 - cx1
                crop_h_px = cy2 - cy1
                
                # Create LandmarkPoint objects (mimics MediaPipe interface)
                landmarks_list = []
                for i in range(468):
                    # Map from crop coords to full frame coords (normalized 0-1)
                    frame_x = (cx1 + raw_lms[i, 0] * crop_w_px) / w
                    frame_y = (cy1 + raw_lms[i, 1] * crop_h_px) / h
                    frame_z = raw_lms[i, 2]
                    landmarks_list.append(LandmarkPoint(frame_x, frame_y, frame_z))
                
                # Velocity-adaptive landmark smoothing (matches MediaPipe internal filter)
                # Heavy smoothing when still (anti-flicker), light when moving (yawn/head turn follows instantly)
                raw_lm_array = np.array([[lp.x, lp.y, lp.z] for lp in landmarks_list], dtype=np.float64)
                if smoothed_landmarks is None:
                    smoothed_landmarks = raw_lm_array.copy()
                else:
                    # Per-landmark velocity: how much each point moved this frame
                    delta = np.abs(raw_lm_array[:, :2] - smoothed_landmarks[:, :2])
                    velocity = np.max(delta, axis=1, keepdims=True)  # [468, 1]
                    # Adaptive alpha: high velocity → alpha~1.0 (follow raw), low velocity → alpha~0.3 (smooth)
                    # Threshold: movement > 0.008 (~4px at 480p) = fast, use raw
                    alpha_per_lm = np.clip(velocity / 0.015, 0.3, 1.0)  # [468, 1]
                    alpha_3d = np.concatenate([alpha_per_lm, alpha_per_lm, alpha_per_lm], axis=1)  # [468, 3]
                    smoothed_landmarks = alpha_3d * raw_lm_array + (1 - alpha_3d) * smoothed_landmarks
                
                landmarks_list = [LandmarkPoint(smoothed_landmarks[i, 0], smoothed_landmarks[i, 1], smoothed_landmarks[i, 2]) for i in range(468)]
                landmarks = landmarks_list
    
    # ---- STEP 3: Run iris landmark on eye crops (with frame-skipping) ----
    iris_left_center = None
    iris_right_center = None
    left_eye_contour = None
    right_eye_contour = None
    
    run_iris_this_frame = (fid % IRIS_SKIP_FRAMES == 0)
    
    if landmarks is not None:
        if run_iris_this_frame:
            # Left eye crop
            left_eye_crop, left_eye_coords = get_eye_crop(frame, landmarks, LEFT_EYE, w, h, margin=0.5)
            if left_eye_crop is not None and left_eye_crop.size > 0:
                ic, left_eye_contour, _ = iris_landmark_detector.predict(left_eye_crop, side=0)
                if ic is not None:
                    ex1, ey1, ex2, ey2 = left_eye_coords
                    iris_left_center = np.array([
                        ex1 + ic[0] * (ex2 - ex1),
                        ey1 + ic[1] * (ey2 - ey1)
                    ])
            
            # Right eye crop
            right_eye_crop, right_eye_coords = get_eye_crop(frame, landmarks, RIGHT_EYE, w, h, margin=0.5)
            if right_eye_crop is not None and right_eye_crop.size > 0:
                ic, right_eye_contour, _ = iris_landmark_detector.predict(right_eye_crop, side=1)
                if ic is not None:
                    ex1, ey1, ex2, ey2 = right_eye_coords
                    iris_right_center = np.array([
                        ex1 + ic[0] * (ex2 - ex1),
                        ey1 + ic[1] * (ey2 - ey1)
                    ])
            
            # Cache results for skipped frames
            cached_iris_left_center = iris_left_center
            cached_iris_right_center = iris_right_center
            cached_left_eye_contour = left_eye_contour
            cached_right_eye_contour = right_eye_contour
        else:
            # Reuse cached iris results
            iris_left_center = cached_iris_left_center
            iris_right_center = cached_iris_right_center
            left_eye_contour = cached_left_eye_contour
            right_eye_contour = cached_right_eye_contour
    
    # ---- STEP 4: Hand detection via Palm Detector + Hand Landmark NPU ----
    detected_hands = []  # list of (hand_lm_list_in_frame_coords, handedness)
    
    run_palm_this_frame = (fid % PALM_SKIP_FRAMES == 0)
    
    if run_palm_this_frame:
        palm_boxes = palm_detector.detect(frame, max_hands=2)
        
        if fid % 30 == 0:
            print(f"[Palm] Raw detections: {len(palm_boxes)} boxes")
            for i, (px1, py1, px2, py2) in enumerate(palm_boxes):
                print(f"  palm[{i}]: ({px1},{py1})-({px2},{py2}) size={px2-px1}x{py2-py1}")
        
        # Filter out false-positive palm boxes:
        # 1. Face detected as palm (centered on face + similar size)
        # 2. Body/chest detected as palm (far below face)
        if face_box is not None and len(palm_boxes) > 0:
            fx1, fy1, fx2, fy2 = face_box
            face_w = fx2 - fx1
            face_h = fy2 - fy1
            face_area = max(1, face_w * face_h)
            face_cx = (fx1 + fx2) / 2
            face_cy = (fy1 + fy2) / 2
            face_bottom = fy2
            filtered_palms = []
            for px1, py1, px2, py2 in palm_boxes:
                palm_cx = (px1 + px2) / 2
                palm_cy = (py1 + py2) / 2
                palm_area = max(1, (px2 - px1) * (py2 - py1))
                
                # FILTER 1: Face-as-palm (centered on face + similar size)
                cdx = abs(palm_cx - face_cx) / max(1, face_w)
                cdy = abs(palm_cy - face_cy) / max(1, face_h)
                size_ratio = palm_area / face_area
                is_centered = cdx < 0.25 and cdy < 0.25
                is_similar_size = 0.5 < size_ratio < 2.0
                if is_centered and is_similar_size:
                    if fid % 30 == 0:
                        print(f"[Palm] Rejected face-as-palm (cdist=({cdx:.2f},{cdy:.2f}), size_ratio={size_ratio:.2f})")
                    continue
                
                # FILTER 2: Body/chest region — reject palms far below face
                # Real hands near face/ear will be at face level or slightly below
                # Body detections are >1.5 face heights below face bottom edge
                below_face_dist = (palm_cy - face_bottom) / max(1, face_h)
                if below_face_dist > 1.0:
                    if fid % 30 == 0:
                        print(f"[Palm] Rejected body-region ({below_face_dist:.2f}x face_h below face)")
                    continue
                
                filtered_palms.append((px1, py1, px2, py2))
            palm_boxes = filtered_palms
            if fid % 30 == 0:
                print(f"[Palm] After filter: {len(palm_boxes)} boxes")
        
        if len(palm_boxes) > 0:
            cached_palm_boxes = palm_boxes
            hand_lost_count = 0
        else:
            hand_lost_count += 1
            if hand_lost_count > MAX_HAND_LOST:
                cached_palm_boxes = []
                cached_hand_landmarks_list = []
    
    # Run hand landmark on detected palms (every frame if palm cached)
    if len(cached_palm_boxes) > 0:
        new_hand_lm_list = []
        for palm_box in cached_palm_boxes:
            px1, py1, px2, py2 = palm_box
            hand_crop = frame[py1:py2, px1:px2]
            if hand_crop is not None and hand_crop.size > 0:
                presence, hand_lms, handedness = hand_landmark_detector.predict(hand_crop)
                if fid % 30 == 0:
                    print(f"[HandLM] palm=({px1},{py1})-({px2},{py2}) presence={presence:.3f} has_lms={hand_lms is not None}")
                if hand_lms is not None and presence > args.hand_presence_threshold:
                    # Convert hand landmarks from crop-relative to frame coords (normalized 0-1)
                    crop_w_px = px2 - px1
                    crop_h_px = py2 - py1
                    frame_hand_lms = []
                    for lm in hand_lms:
                        fx = (px1 + lm.x * crop_w_px) / w
                        fy = (py1 + lm.y * crop_h_px) / h
                        fz = lm.z
                        frame_hand_lms.append(LandmarkPoint(fx, fy, fz))
                    new_hand_lm_list.append((frame_hand_lms, handedness))
        
        if len(new_hand_lm_list) > 0:
            cached_hand_landmarks_list = new_hand_lm_list
        else:
            # Hand landmark model rejected ALL crops — clear both caches
            # This prevents stale palm boxes from retrying HandLM every frame
            cached_hand_landmarks_list = []
            cached_palm_boxes = []
            hand_lost_count = 0
        detected_hands = cached_hand_landmarks_list
    
    # FPS calculation
    frame_end = time.time()
    frame_ms = (frame_end - frame_start) * 1000
    fps_frame_count += 1
    elapsed = frame_end - fps_start_time
    if elapsed >= 1.0:
        fps = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_start_time = frame_end

    # Log detection status every 30 frames
    if fid % 30 == 0:
        face_status = f"Face detected (conf={face_confidence:.2f})" if landmarks else "No face"
        hand_status = f"{len(detected_hands)} hand(s)" if detected_hands else "No hand"
        print(f"[Frame {fid}] {face_status}, {hand_status} | FPS: {fps:.1f} | Frame: {frame_ms:.1f}ms")

    # ---- DMS LOGIC (same as original dmsv8.py) ----
    if landmarks is not None:
        face_center = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))
        
        # EYE CLOSURE (with adaptive smoothing — matches MediaPipe ref)
        left_ear = get_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = get_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        
        visible_ears = []
        if left_ear > 0: visible_ears.append(left_ear)
        if right_ear > 0: visible_ears.append(right_ear)
        raw_avg_ear = np.mean(visible_ears) if visible_ears else 1.0
        
        # Adaptive EAR smoothing (fast close, slow open — matches MediaPipe ref)
        if smoothed_ear is None:
            smoothed_ear = raw_avg_ear
        else:
            if raw_avg_ear < smoothed_ear:  # Eyes closing
                SMOOTH_FACTOR = 0.6  # Fast response (60% new)
            else:  # Eyes opening
                SMOOTH_FACTOR = 0.3  # Slower (30% new) — reduce false clears
            smoothed_ear = SMOOTH_FACTOR * raw_avg_ear + (1 - SMOOTH_FACTOR) * smoothed_ear
        
        avg_ear = smoothed_ear
        
        msg = last_msg
        msg = "Normal and Active Driving"
        eye_closed = 0
        head_turn = 0
        hands_free = False
        head_tilt = 0
        head_droop = 0
        yawn_flag = False

        # Iris visibility check using iris model results
        iris_visible = (iris_left_center is not None) or (iris_right_center is not None)
        
        # Compute iris center average for gaze
        if iris_left_center is not None and iris_right_center is not None:
            iris_center_avg = (iris_left_center + iris_right_center) / 2
        elif iris_left_center is not None:
            iris_center_avg = iris_left_center
        elif iris_right_center is not None:
            iris_center_avg = iris_right_center
        else:
            # Fallback: estimate from eye corner midpoints (468-point model has no iris landmarks)
            iris_center_avg = get_iris_center_from_landmarks(landmarks, LEFT_IRIS_FALLBACK + RIGHT_IRIS_FALLBACK, w, h)

        iris_y_avg = iris_center_avg[1] / h if iris_center_avg is not None else 0.5
        iris_missing_or_low = (not iris_visible) or (iris_y_avg > 0.5)
        
        eye_closed_by_ear = avg_ear < args.ear_threshold
        raw_eye_closed = raw_avg_ear < args.ear_threshold
        
        # Calculate eye openness percentage (matches MediaPipe ref)
        EAR_FULLY_OPEN = 0.28
        EAR_FULLY_CLOSED = 0.10
        if avg_ear <= EAR_FULLY_CLOSED:
            current_eye_openness = 0.0
        elif avg_ear >= EAR_FULLY_OPEN:
            current_eye_openness = 100.0
        else:
            current_eye_openness = ((avg_ear - EAR_FULLY_CLOSED) / (EAR_FULLY_OPEN - EAR_FULLY_CLOSED)) * 100.0
        
        # Eye closure duration tracking (matches MediaPipe ref)
        now_eye = time.time()
        if eye_closed_by_ear:
            if eye_close_start_time is None:
                eye_close_start_time = now_eye
            eye_close_duration = now_eye - eye_close_start_time
            if eye_close_duration >= 3.0:
                add_alert(frame, f"Eye Closure: {eye_close_duration:.1f}s CRITICAL")
            elif eye_close_duration >= 2.0 and fid % 5 == 0:
                add_alert(frame, f"Eye Closure: {eye_close_duration:.1f}s WARNING")
        
        # Use RAW EAR for reset (faster response — matches MediaPipe ref)
        if not eye_closed_by_ear or (not raw_eye_closed and eye_close_start_time is not None):
            if eye_close_start_time is not None:
                final_dur = now_eye - eye_close_start_time
                if final_dur > 0.5:
                    print(f"[EyeTimer] Reopened after {final_dur:.1f}s")
                # Force reset smoothed EAR to raw (eliminate lag)
                if raw_avg_ear > args.ear_threshold + 0.02:
                    smoothed_ear = raw_avg_ear
            eye_close_start_time = None
        
        if eye_closed_by_ear and iris_missing_or_low:
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
                blink_total += 1
            eye_closure_counter = 0
        
        # Track eye state transitions (matches MediaPipe ref)
        curr_eye_state = 'closed' if (eye_closed_by_ear and iris_missing_or_low) else 'open'
        if prev_eye_state is not None and curr_eye_state != prev_eye_state:
            print(f"[Eye] {prev_eye_state} -> {curr_eye_state}")
        prev_eye_state = curr_eye_state

        if current_time - blink_timer > 60:
            if blink_counter >= args.blink_rate_threshold:
                msg = "High Blinking Rate"
                msg = add_alert(frame, msg)
                last_msg = msg
            blink_counter = 0
            blink_timer = current_time

        # YAWN DETECTION
        mar = get_mar(landmarks, MOUTH, w, h)
        mar_deque.append(mar)
        if mar > args.mar_threshold:
            yawn_counter += 1

        if yawn_counter > args.yawn_threshold:
            msg = "Warning: Yawning"
            msg = add_alert(frame, msg)
            last_msg = msg
            yawn_flag = True
            yawn_counter = 0

        # GAZE ESTIMATION (using iris model output)
        gaze_x_norm = iris_center_avg[0] / w if iris_center_avg is not None else 0.5

        # HEAD POSE (using nose tip with EMA smoothing — matches MediaPipe ref)
        raw_head_x = landmarks[NOSE_TIP].x
        raw_head_y = landmarks[NOSE_TIP].y
        alpha = 0.3  # EMA factor (matches MediaPipe ref)
        
        if calibration_mode:
            # During calibration, collect samples (matches MediaPipe ref median-based)
            head_x = raw_head_x
            head_y = raw_head_y
            cv2.putText(frame, "Please align your face naturally, facing forward, before the countdown ends", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            # Smoothed head position (matches MediaPipe ref)
            dx = raw_head_x - head_center_x
            dy = raw_head_y - head_center_y
            sm_head_x_signed = alpha * dx + (1 - alpha) * sm_head_x_signed
            sm_head_y_signed = alpha * dy + (1 - alpha) * sm_head_y_signed
            head_x = raw_head_x
            head_y = raw_head_y
            
            # Adaptive head recalibration (drift compensation — matches MediaPipe ref)
            eyes_open_forward = (abs(sm_head_x_signed) < 0.05) and (abs(sm_head_y_signed) < 0.05) and (prev_eye_state != 'closed')
            if eyes_open_forward:
                beta = 0.05
                head_center_x = (1 - beta) * head_center_x + beta * head_x
                head_center_y = (1 - beta) * head_center_y + beta * head_y
            
            # GAZE DETECTION with confirmation streaks (matches MediaPipe ref)
            gaze_deviation = gaze_x_norm - gaze_center  # negative=left, positive=right
            now_gaze = time.time()
            
            LEFT_THRESHOLD = args.gaze_deviation_threshold   # 0.025 (matches ref)
            RIGHT_THRESHOLD = args.gaze_deviation_threshold  # 0.025 symmetric
            
            if gaze_deviation < -LEFT_THRESHOLD:
                gaze_left_confirm_streak += 1
                gaze_right_confirm_streak = 0
                if gaze_left_confirm_streak >= GAZE_CONFIRM_FRAMES:
                    if not gaze_alerted_left and (now_gaze - last_gaze_left_alert_time >= GAZE_ALERT_COOLDOWN):
                        add_alert(frame, "Looking Right")
                        gaze_alerted_left = True
                        last_gaze_left_alert_time = now_gaze
            elif gaze_deviation > RIGHT_THRESHOLD:
                gaze_right_confirm_streak += 1
                gaze_left_confirm_streak = 0
                if gaze_right_confirm_streak >= GAZE_CONFIRM_FRAMES:
                    if not gaze_alerted_right and (now_gaze - last_gaze_right_alert_time >= GAZE_ALERT_COOLDOWN):
                        add_alert(frame, "Looking Left")
                        gaze_alerted_right = True
                        last_gaze_right_alert_time = now_gaze
            else:
                # Reset when gaze returns to center
                gaze_left_confirm_streak = 0
                gaze_right_confirm_streak = 0
                gaze_alerted_left = False
                gaze_alerted_right = False
            
            # HEAD TURN detection using smoothed offsets (matches MediaPipe ref)
            sm_head_x_offset = abs(sm_head_x_signed)
            sm_head_y_offset = abs(sm_head_y_signed)
            
            # Check vertical FIRST (higher priority — matches MediaPipe ref)
            if sm_head_y_signed < 0:  # HEAD UPWARD
                if sm_head_y_offset >= 0.015:
                    head_tilt = 1
                    add_alert(frame, "Looking Upward")
            elif sm_head_y_signed > 0:  # HEAD DROOP/DOWNWARD
                if sm_head_y_offset >= 0.05:
                    head_droop = 1
                    add_alert(frame, "Head Downward")
            
            # Horizontal turn (only if no vertical detected — matches MediaPipe ref)
            if head_tilt == 0 and head_droop == 0:
                if sm_head_x_offset > args.head_turn_threshold:
                    head_turn = 1
                    direction = "Left" if sm_head_x_signed < 0 else "Right"
                    add_alert(frame, f"Head Turn {direction}")

        # Draw face mesh (landmarks) if enabled
        if not args.no_mesh_display and landmarks is not None:
            for lm in landmarks:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)
            
            # Draw iris centers
            if iris_left_center is not None:
                cv2.circle(frame, (int(iris_left_center[0]), int(iris_left_center[1])), 3, (255, 0, 255), -1)
            if iris_right_center is not None:
                cv2.circle(frame, (int(iris_right_center[0]), int(iris_right_center[1])), 3, (255, 0, 255), -1)

    # ---- Hand detection alerts (with confirmation streaks to prevent false positives) ----
    frame_hand_near_ear = False
    frame_hand_near_face = False
    frame_texting = False
    
    if len(detected_hands) > 0 and landmarks is not None:
        hand_coords = []
        for hand_lm_list, handedness in detected_hands:
            # Check phone call: require BOTH near ear AND near face (matches ref)
            near_ear = hand_near_ear(landmarks, hand_lm_list, w, h)
            near_face = hand_near_face(face_center, hand_lm_list, w, h, args.hand_near_face_px)
            
            if near_ear and near_face:
                frame_hand_near_ear = True
            elif near_face:
                frame_hand_near_face = True
            
            if fid % 30 == 0:
                hcx = np.mean([lm.x for lm in hand_lm_list])
                hcy = np.mean([lm.y for lm in hand_lm_list])
                print(f"[Hand] center=({hcx:.2f},{hcy:.2f}) near_ear={near_ear} near_face={near_face} handedness={handedness:.2f}")
            
            # Collect hand center for texting detection
            xs = [lm.x for lm in hand_lm_list]
            ys = [lm.y for lm in hand_lm_list]
            hand_coords.append((np.mean(xs), np.mean(ys)))
            
            # Draw hand landmarks MediaPipe-style (skeleton + colored joints)
            if not args.no_mesh_display:
                draw_hand_landmarks_mp(frame, hand_lm_list, w, h)
        
        # Texting detection: 2 hands, both low, close together (matches dmsv8)
        if not calibration_mode and len(hand_coords) == 2:
            (x1h, y1h), (x2h, y2h) = hand_coords
            dist = np.hypot(x2h - x1h, y2h - y1h)
            both_hands_low = y1h > 0.6 and y2h > 0.6
            not_near_ears = True
            for hand_lm_list, _ in detected_hands:
                if hand_near_ear(landmarks, hand_lm_list, w, h):
                    not_near_ears = False
                    break
            if dist < 0.35 and both_hands_low and not_near_ears:
                frame_texting = True
    
    # Confirmation streaks: only alert after N consecutive frames (prevents random false positives)
    if frame_hand_near_ear:
        hand_near_ear_streak += 1
        if hand_near_ear_streak >= HAND_CONFIRM_FRAMES:
            msg = "Likely mobile call"
            msg = add_alert(frame, msg)
            last_msg = msg
            hands_free = True
    else:
        hand_near_ear_streak = 0
    
    if frame_hand_near_face:
        hand_near_face_streak += 1
        if hand_near_face_streak >= HAND_CONFIRM_FRAMES:
            msg = "Hand near the face"
            msg = add_alert(frame, msg)
            last_msg = msg
            hands_free = True
    else:
        hand_near_face_streak = 0
    
    if frame_texting:
        texting_streak += 1
        if texting_streak >= HAND_CONFIRM_FRAMES:
            msg = "Possible texting observed"
            msg = add_alert(frame, msg)
            last_msg = msg
            hands_free = True
    else:
        texting_streak = 0

    # Combined drowsiness detection
    if eye_closed == 2 and head_droop >= 1 or eye_closed == 2 and yawn_flag:
        msg = "Severe DROWSINESS Observed"
        msg = add_alert(frame, msg)
        last_msg = msg
    elif eye_closed == 1 and head_droop >= 1 or eye_closed == 1 and yawn_flag:
        msg = "Moderate DROWSINESS Observed"
        msg = add_alert(frame, msg)
        last_msg = msg
    
    # Combined distraction detection (matches dmsv8)
    if head_turn >= 1 and hands_free or head_tilt >= 1 and hands_free:
        msg = "Moderate DISTRACTION Observed"
        msg = add_alert(frame, msg)
        last_msg = msg

    # Expire alerts
    expired = [k for k, t in active_alerts.items() if current_time - t > ALERT_DURATION]
    for k in expired:
        del active_alerts[k]

    for i, msg in enumerate(active_alerts):
        if "Mild" in msg or "Warning" in msg:
            color = (255, 255, 255)
        elif "Moderate" in msg or "Alert" in msg:
            color = (0, 255, 255)
        elif "Severe" in msg:
            color = (0, 0, 255)
        else:
            color = (0, 0, 255)
        cv2.putText(frame, msg, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw FPS on frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Update MJPEG stream
    if mjpeg_server is not None:
        mjpeg_server.update_frame(frame)

    if not args.headless:
        cv2.imshow("Drowsiness Monitor", frame)
    
    # Collect calibration samples during countdown (matches MediaPipe ref median-based)
    if calibration_mode and landmarks is not None:
        calib_gaze_x_samples.append(float(gaze_x_norm))
        calib_head_x_samples.append(float(landmarks[NOSE_TIP].x))
        calib_head_y_samples.append(float(landmarks[NOSE_TIP].y))
    
    if calibration_mode:
        countdown = int(calibration_duration - (time.time() - calibration_start_time))
        if countdown < 0:
            countdown = 0
        if countdown > 0:
            if countdown % 5 == 0 and fid % 30 == 0:
                if landmarks:
                    print(f"[Calibration] {countdown}s remaining - Face detected OK (samples: {len(calib_gaze_x_samples)})")
                else:
                    print(f"[Calibration] {countdown}s remaining - WARNING: No face detected!")
            cv2.putText(frame, f"{countdown}s", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            if not args.headless:
                cv2.imshow("Drowsiness Monitor", frame)
            key = cv2.waitKey(1) & 0xFF if not args.headless else 0xFF
        if countdown == 0 and calibration_mode:
            # Use MEDIAN of collected samples for robust baseline (matches MediaPipe ref)
            if len(calib_gaze_x_samples) >= calib_min_samples:
                gaze_center = float(np.median(list(calib_gaze_x_samples)))
                head_center_x = float(np.median(list(calib_head_x_samples)))
                head_center_y = float(np.median(list(calib_head_y_samples)))
                calibration_mode = False
                print(f"[Calibration] COMPLETE! Gaze center LOCKED at {gaze_center:.3f}")
                print(f"[Calibration] Samples: {len(calib_gaze_x_samples)}, Head center: ({head_center_x:.3f}, {head_center_y:.3f})")
                add_alert(frame, "Calibration Complete")
            elif landmarks is not None:
                # Fallback to single-frame if insufficient samples
                gaze_center = gaze_x_norm
                head_center_x = landmarks[NOSE_TIP].x
                head_center_y = landmarks[NOSE_TIP].y
                calibration_mode = False
                print(f"[Calibration] COMPLETE (fallback)! Gaze center: {gaze_center:.3f}")
                add_alert(frame, "Calibration Complete")
            else:
                calibration_mode = False
                print(f"[Calibration] WARNING - Completed without face detection! Using defaults.")
    
    if not args.headless:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    else:
        time.sleep(0.001)

print("\n[DMS] Shutting down...")
cap.release()
if not args.headless:
    cv2.destroyAllWindows()
if mjpeg_server is not None:
    mjpeg_server.stop()
