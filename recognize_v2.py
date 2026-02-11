"""
Face Recognition System - Industrial Grade
Uses SCRFD for detection + 5-point alignment for robust face preprocessing
Based on NXP reference implementation
"""

import cv2
import numpy as np
import json
import time
from collections import deque
import tensorflow as tf
from scrfd_detector import SCRFDDetector, align_face

class FaceRecognizer:
    def __init__(self, detector_path, recognizer_path, database_path):
        """Initialize face detector and recognizer"""
        # Load SCRFD face detector with landmark detection
        self.detector = SCRFDDetector(detector_path)
        
        # Load Facenet recognizer
        self.recognizer = tf.lite.Interpreter(model_path=recognizer_path)
        self.recognizer.allocate_tensors()
        self.recognizer_input = self.recognizer.get_input_details()[0]
        self.recognizer_output = self.recognizer.get_output_details()[0]
        
        # Get input shape for recognizer - handle channels-first format
        full_shape = self.recognizer_input['shape']
        print(f"Full input shape: {full_shape}")
        
        if full_shape[1] == 3:  # Channels-first: [1, 3, 112, 112]
            self.input_shape = (int(full_shape[2]), int(full_shape[3]))  # (112, 112)
            self.channels_first = True
        else:  # Channels-last: [1, 112, 112, 3]
            self.input_shape = (int(full_shape[1]), int(full_shape[2]))  # (112, 112)
            self.channels_first = False
        
        print(f"Recognizer input shape (HxW): {self.input_shape}, Channels-first: {self.channels_first}")
        
        # Load enrolled profiles
        self.enrolled_profiles = self.load_database(database_path)
        print(f"Loaded {len(self.enrolled_profiles)} enrolled profiles")
        
        # Recognition parameters - STRICT to prevent false positives
        self.similarity_threshold = 0.80  # High threshold - must be very similar
        self.strict_threshold = 0.90     # Very strict for high confidence
        self.margin_threshold = 0.07      # Clear margin required between best and 2nd
        self.consecutive_frames = 2       # Require 3 consecutive frames for stability
        
        # Multi-frame averaging for robustness
        self.embedding_history = deque(maxlen=5)  # Store last 5 embeddings
        self.use_averaging = True  # Enable embedding averaging
        
        # State tracking
        self.current_person = None
        self.match_streak = 0
        self.last_match_time = time.time()
        
        # Lock-in state - once recognized, stop running model until face disappears
        self.locked_driver = None  # Locked driver name
        self.locked_similarity = 0.0  # Locked similarity score
        self.no_face_frames = 0  # Counter for frames without face
        self.no_face_threshold = 10  # Frames without face before unlocking
        
        # Unrecognized tracking - stop model if person not recognized after threshold
        self.unrecognized_frames = 0  # Counter for unrecognized frames
        self.unrecognized_threshold = 15  # Frames without recognition before stopping
        self.should_register = False  # Flag to show "Please Register" message
        
        # Frame skipping for efficiency (only used during active recognition)
        self.frame_counter = 0
        self.embedding_check_interval = 10  # Check embedding every 10 frames
        self.last_recognized_name = "Unknown"
        self.last_similarity = 0.0

    def load_database(self, path):
        """Load enrolled profiles from drivers.json"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            profiles = {}
            for profile in data.get('profiles', []):
                name = profile['name']
                embedding = np.array(profile['embedding'], dtype=np.float32)
                # Normalize to unit vector
                embedding = embedding / np.linalg.norm(embedding)
                profiles[name] = {
                    'embedding': embedding,
                    'driver_id': profile.get('driver_id', len(profiles) + 1)  # Handle missing driver_id
                }
                print(f"  Loaded: {name} (ID: {profiles[name]['driver_id']})")
            return profiles
        except Exception as e:
            print(f"Error loading database: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def detect_faces(self, frame):
        """Detect faces using SCRFD with landmarks - STRICT to avoid false positives"""
        detections = self.detector.detect(frame, score_threshold=0.75)  # Raised from 0.45 to 0.70 for strict detection
        return detections

    def _parse_yolo_output(self, output, frame_shape):
        """Parse YOLO grid output - OFFICIAL NXP METHOD"""
        h, w = frame_shape[:2]
        
        # Dequantize using official NXP formula
        scale, zero_point = self.detector_output['quantization']
        output = output[0].astype(np.float32)
        output = (output + 15) * 0.14218327403068542  # Official NXP dequantization
        
        # Reshape to [7, 7, 3 anchors, 6 values]
        output = output.reshape((7, 7, 3, 6)).transpose([2, 0, 1, 3])
        
        # Official NXP anchor boxes
        anchors = np.zeros([3, 1, 1, 2], dtype=np.float32)
        anchors[0, 0, 0, :] = [9, 14]
        anchors[1, 0, 0, :] = [12, 17]
        anchors[2, 0, 0, :] = [22, 21]
        
        # Create grid
        yv, xv = np.meshgrid(np.arange(7), np.arange(7))
        grid = np.stack((yv, xv), 2).reshape((1, 7, 7, 2)).astype(np.float32)
        
        # Decode boxes: (sigmoid(xy) + grid) * 8
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        output[..., 0:2] = (sigmoid(output[..., 0:2]) + grid) * 8
        output[..., 2:4] = np.exp(output[..., 2:4]) * anchors
        output[..., 4:] = sigmoid(output[..., 4:])
        
        # Filter by confidence threshold
        prediction = output.reshape((-1, 6))
        x = prediction[prediction[..., 4] > 0.75]  # Lower threshold for better detection
        
        if not x.shape[0]:
            return []
        
        detections = []
        for box in x:
            # box format: [cx, cy, w, h, conf, class]
            cx, cy, bw, bh, conf = box[0], box[1], box[2], box[3], box[4]
            
            # Convert to normalized coordinates
            x1 = (cx - bw / 2) / 56.0  # input size is 56
            y1 = (cy - bh / 2) / 56.0
            x2 = (cx + bw / 2) / 56.0
            y2 = (cy + bh / 2) / 56.0
            
            # Convert to pixel coordinates
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            
            # Clip to frame bounds
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Size validation
            box_w = x2 - x1
            box_h = y2 - y1
            rel_w = box_w / w
            rel_h = box_h / h
            
            if rel_w < self.min_face_size or rel_h < self.min_face_size:
                continue  # Too small
            if rel_w > self.max_face_size or rel_h > self.max_face_size:
                continue  # Too large
            
            # Aspect ratio validation
            aspect_ratio = box_h / box_w
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue  # Wrong shape
            
            detections.append([x1, y1, x2, y2, conf])
        
        # Apply NMS
        detections = self._nms(detections, iou_threshold=0.3)
        
        # Temporal smoothing with exponential moving average
        if len(detections) > 0:
            current_box = detections[0]
            
            if len(self.detection_history) > 0:
                # Smooth with previous detection
                prev_box = self.detection_history[-1]
                alpha = 0.7  # Smoothing factor
                smoothed = [
                    int(alpha * current_box[0] + (1-alpha) * prev_box[0]),
                    int(alpha * current_box[1] + (1-alpha) * prev_box[1]),
                    int(alpha * current_box[2] + (1-alpha) * prev_box[2]),
                    int(alpha * current_box[3] + (1-alpha) * prev_box[3]),
                    current_box[4]
                ]
                current_box = smoothed
            
            self.detection_history.append(current_box)
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            
            # Need at least min_stable_frames for stability
            if len(self.detection_history) >= self.min_stable_frames:
                return [current_box]
            else:
                return []  # Not stable yet
        else:
            # Gradually decay history instead of clearing
            if len(self.detection_history) > 0:
                self.detection_history.pop(0)
            return []

    def _nms(self, detections, iou_threshold=0.3):
        """Non-maximum suppression"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        scores = np.array([d[4] for d in detections])
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]

    def extract_embedding(self, aligned_face):
        """Extract face embedding with ROBUST multi-stage lighting normalization"""
        # STAGE 1: Gamma correction for exposure normalization
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Adaptive gamma based on brightness
        if mean_brightness < 80:  # Dark image
            gamma = 1.5  # Brighten
        elif mean_brightness > 175:  # Bright image
            gamma = 0.7  # Darken
        else:
            gamma = 1.0  # No adjustment
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            aligned_face = cv2.LUT(aligned_face, table)
        
        # STAGE 2: Multi-scale Retinex for color constancy
        def single_scale_retinex(img, sigma):
            retinex = np.log10(img.astype(np.float32) + 1.0) - np.log10(cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma) + 1.0)
            return retinex
        
        # Apply MSR on each channel
        img_msr = np.zeros_like(aligned_face, dtype=np.float32)
        for i in range(3):
            msr_channel = (single_scale_retinex(aligned_face[:,:,i], 15) + 
                          single_scale_retinex(aligned_face[:,:,i], 80) + 
                          single_scale_retinex(aligned_face[:,:,i], 200)) / 3.0
            img_msr[:,:,i] = msr_channel
        
        # Normalize MSR output to [0, 255]
        img_msr = (img_msr - np.min(img_msr)) / (np.max(img_msr) - np.min(img_msr) + 1e-6) * 255.0
        img_msr = np.clip(img_msr, 0, 255).astype(np.uint8)
        
        # STAGE 3: CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
        img_eq = np.zeros_like(img_msr, dtype=np.uint8)
        for i in range(3):
            img_eq[:,:,i] = clahe.apply(img_msr[:,:,i])
        
        # Convert to RGB and float
        img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # STAGE 4: Fixed standardization (lighting invariant)
        img_normalized = (img_rgb - 127.5) / 128.0
        
        # Quantize to INT8 using model's quantization parameters
        scale = self.recognizer_input['quantization'][0]
        zero_point = self.recognizer_input['quantization'][1]
        
        img_int8 = (img_normalized / scale + zero_point).astype(np.int8)
        
        # Add batch dimension and transpose to channels-first [1, 3, 112, 112]
        input_data = np.expand_dims(img_int8, axis=0)
        if self.channels_first:
            input_data = np.transpose(input_data, (0, 3, 1, 2))
        
        # Run inference
        self.recognizer.set_tensor(self.recognizer_input['index'], input_data)
        self.recognizer.invoke()
        embedding = self.recognizer.get_tensor(self.recognizer_output['index'])[0]
        
        # Dequantize output: float = (int8 - zero) * scale
        out_scale, out_zero = self.recognizer_output['quantization']
        embedding = (embedding.astype(np.float32) - out_zero) * out_scale
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

    def recognize_face(self, embedding):
        """Match embedding against enrolled profiles with strict validation"""
        if len(self.enrolled_profiles) == 0:
            return "Unknown", 0.0, False
        
        # Add to embedding history for averaging
        self.embedding_history.append(embedding)
        
        # Use averaged embedding for more stable recognition
        if self.use_averaging and len(self.embedding_history) >= 3:
            avg_embedding = np.mean(list(self.embedding_history), axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            test_embedding = avg_embedding
        else:
            test_embedding = embedding
        
        # Compute similarities
        similarities = {}
        for name, profile in self.enrolled_profiles.items():
            enrolled_emb = profile['embedding']
            similarity = np.dot(test_embedding, enrolled_emb)
            similarities[name] = similarity
        
        # Sort by similarity
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_name, best_sim = sorted_matches[0]
        second_best_sim = sorted_matches[1][1] if len(sorted_matches) > 1 else 0.0
        
        margin = best_sim - second_best_sim
        
        # STRICT VALIDATION - Multiple checks to prevent false positives (no printing)
        
        # Check 1: Main threshold
        if best_sim < self.similarity_threshold:
            self.current_person = None
            self.match_streak = 0
            # Don't clear history - allow recovery
            return "Unknown", best_sim, False
        
        # Check 2: Strict threshold for high confidence
        if best_sim < self.strict_threshold:
            # Don't reject immediately - check consistency first
            pass
        
        # Check 3: Margin validation - must be clearly better than second best
        # This is a warning, not a hard rejection - allow it to proceed if consistency is good
        if len(self.enrolled_profiles) > 1 and margin < self.margin_threshold:
            # Don't reject immediately - check consistency instead
            # Only reject if BOTH margin is low AND we don't have consistent history
            if len(self.embedding_history) < 3:
                self.current_person = None
                self.match_streak = 0
                return "Unknown", best_sim, False
        
        # Check 4: Consistency across embedding history (only if we have enough samples)
        if len(self.embedding_history) >= 4:  # Changed from 3 to 4 - need more samples
            # Check if all recent embeddings agree
            recent_sims = []
            for hist_emb in list(self.embedding_history)[-4:]:
                sim = np.dot(hist_emb, self.enrolled_profiles[best_name]['embedding'])
                recent_sims.append(sim)
            
            sim_std = np.std(recent_sims)
            sim_min = np.min(recent_sims)
            sim_mean = np.mean(recent_sims)
            
            # Reject if embeddings are VERY inconsistent (high variance)
            if sim_std > 0.10:  # Strict consistency required
                self.current_person = None
                self.match_streak = 0
                self.embedding_history.clear()  # Clear on high inconsistency
                return "Unknown", best_sim, False
            
            # Reject if minimum recent score is too low
            if sim_min < 0.70:  # Strict minimum threshold
                self.current_person = None
                self.match_streak = 0
                # Don't clear history - allow recovery
                return "Unknown", best_sim, False
            
            # If consistency is good, override strict threshold check
            if sim_std < 0.04 and sim_mean >= self.similarity_threshold:
                # Allow recognition even if current frame is below strict threshold
                pass
        
        # Consecutive frame validation
        if self.current_person == best_name:
            self.match_streak += 1
            self.last_match_time = time.time()
        else:
            # Changing person - reset everything
            self.current_person = best_name
            self.match_streak = 1
            self.last_match_time = time.time()
        
        # Check 5: Require consecutive frames for stability
        if self.match_streak >= self.consecutive_frames:
            return best_name, best_sim, True
        else:
            return "Unknown", best_sim, False



    def run(self):
        """Main recognition loop with lock-in mechanism"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n=== Face Recognition Started ===")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            detections = self.detect_faces(frame)
            
            # Check if face is present
            if len(detections) == 0:
                # No face detected
                self.no_face_frames += 1
                
                # If no face for threshold frames, unlock
                if self.no_face_frames >= self.no_face_threshold and self.locked_driver:
                    print(f"🔓 Driver {self.locked_driver} left - unlocking")
                    self.locked_driver = None
                    self.locked_similarity = 0.0
                    self.current_person = None
                    self.match_streak = 0
                    self.embedding_history.clear()
                    self.unrecognized_frames = 0
                    self.should_register = False
                
                # Display message
                cv2.putText(frame, "No Face Detected", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if self.locked_driver:
                    cv2.putText(frame, f"Last Driver: {self.locked_driver}", (20, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # Face detected - reset no-face counter
                self.no_face_frames = 0
                
                det = detections[0]  # Use first (best) detection
                box = det['box']
                keypoints = det['keypoints']
                score = det['score']
                
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # If driver is locked, just display the locked result
                if self.locked_driver:
                    name = self.locked_driver
                    color = (0, 255, 0)  # Green
                    label = f"{name}"  # No threshold display
                elif self.should_register:
                    # Exceeded unrecognized threshold - stop recognition
                    name = "Unknown"
                    color = (0, 0, 255)  # Red
                    label = "Please Register"
                else:
                    # Not locked - run recognition
                    self.frame_counter += 1
                    
                    # Only compute embedding every N frames during recognition phase
                    if self.frame_counter % self.embedding_check_interval == 0:
                        # Align face using 5-point landmarks
                        aligned_face = align_face(frame, keypoints, target_size=self.input_shape)
                        
                        # Extract embedding from aligned face
                        embedding = self.extract_embedding(aligned_face)
                        name, similarity, is_stable = self.recognize_face(embedding)
                        
                        # Update last recognized state
                        if is_stable and name != "Unknown":
                            self.last_recognized_name = name
                            self.last_similarity = similarity
                            
                            # Lock the driver once stable recognition is achieved
                            self.locked_driver = name
                            self.locked_similarity = similarity
                            self.unrecognized_frames = 0
                            print(f"🔒 LOCKED: {name}")
                        else:
                            self.last_recognized_name = "Unknown"
                            self.last_similarity = 0.0
                            self.unrecognized_frames += 1
                            
                            # Check if exceeded unrecognized threshold
                            if self.unrecognized_frames >= self.unrecognized_threshold:
                                self.should_register = True
                                print(f"⚠ Person not recognized after {self.unrecognized_threshold} frames - Please register")
                    else:
                        # Use previous recognition result
                        name = self.last_recognized_name
                        similarity = self.last_similarity
                        is_stable = (name != "Unknown")
                    
                    # Draw results
                    if is_stable and name != "Unknown":
                        color = (0, 255, 0)  # Green for recognized
                        label = f"{name}"  # No threshold display
                    else:
                        color = (0, 165, 255)  # Orange for recognizing
                        label = "Recognizing..."
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw landmarks
                kps = keypoints.reshape(5, 2).astype(int)
                for kx, ky in kps:
                    cv2.circle(frame, (kx, ky), 2, (0, 255, 255), -1)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n=== Recognition Stopped ===")


if __name__ == "__main__":
    recognizer = FaceRecognizer(
        detector_path="scrfd_500m_full_int8.tflite",
        recognizer_path="fr_int8.tflite",
        database_path="drivers.json"
    )
    recognizer.run()