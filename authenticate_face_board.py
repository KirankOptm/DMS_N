"""
Face Authentication System - IMX Board NPU Version
Authenticates enrolled drivers using vela-optimized models on NPU
Uses same enrollment database as enroll_industrial_board.py
"""

import cv2
import numpy as np
import json
import time
import os
from collections import deque

# Latency tracker for face recognition pipeline
try:
    import dms_latency_tracker as lat
    _LAT = True
    print("[FaceAuth] Latency tracking ENABLED")
except ImportError:
    _LAT = False
    print("[FaceAuth] Latency tracking DISABLED (dms_latency_tracker not found)")

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("[WARNING] tflite_runtime not found, falling back to tensorflow.lite")
    import tensorflow.lite as tflite

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    print("[WARNING] MediaPipe not found - liveness detection will use basic mode")
    _HAS_MEDIAPIPE = False

from scrfd_detector_board import SCRFDDetector, align_face


class LivenessDetector:
    """
    Anti-Spoofing Liveness Detection
    Prevents authentication using photos or videos
    
    Implements two approaches:
    1. Eye Blink Detection (simple, low CPU)
    2. Micro-Motion Analysis (fool-proof, slightly higher CPU)
    """
    def __init__(self, blink_threshold=2, motion_threshold=0.8, buffer_size=15):
        print("\n[Liveness] Initializing anti-spoofing detection...")
        
        # Thresholds
        self.MIN_BLINKS = blink_threshold        # Minimum blinks required
        self.MICRO_MOTION_THRESH = motion_threshold  # Micro-motion threshold
        self.BUFFER_SIZE = buffer_size
        
        # Eye blink detection (EAR - Eye Aspect Ratio)
        self.EAR_THRESHOLD = 0.21
        self.EAR_CONSEC_FRAMES = 2
        self.blink_counter = 0
        self.ear_counter = 0
        
        # Head pose variation
        self.pose_history = deque(maxlen=30)
        self.initial_pose = None
        
        # Micro-motion buffer
        self.micro_motion_buffer = deque(maxlen=buffer_size)
        self.prev_landmarks = None
        
        # State
        self.has_blinked = False
        self.has_head_movement = False
        self.has_micro_motion = False
        
        # MediaPipe Face Mesh
        if _HAS_MEDIAPIPE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[Liveness] ✓ MediaPipe Face Mesh initialized (full mode)")
        else:
            self.face_mesh = None
            print("[Liveness] ⚠ MediaPipe unavailable - using SCRFD keypoints only")
        
        print(f"[Liveness] Blink threshold: {self.MIN_BLINKS} blinks")
        print(f"[Liveness] Micro-motion threshold: {self.MICRO_MOTION_THRESH}")
        print(f"[Liveness] Buffer size: {self.BUFFER_SIZE} frames")
    
    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # Compute euclidean distances between vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Compute euclidean distance between horizontal eye landmarks
        C = np.linalg.norm(eye[0] - eye[3])
        # EAR formula
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def detect_blink_scrfd(self, keypoints):
        """Detect blinks using SCRFD 5-point landmarks (basic mode)"""
        # SCRFD keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = keypoints[0]
        right_eye = keypoints[1]
        
        # Approximate EAR using eye-to-nose distance (simple vertical distance)
        nose = keypoints[2]
        
        left_eye_dist = abs(left_eye[1] - nose[1])
        right_eye_dist = abs(right_eye[1] - nose[1])
        
        # Normalize by inter-eye distance
        eye_dist = np.linalg.norm(left_eye - right_eye)
        left_ratio = left_eye_dist / (eye_dist + 1e-6)
        right_ratio = right_eye_dist / (eye_dist + 1e-6)
        
        avg_ratio = (left_ratio + right_ratio) / 2.0
        
        # Detect blink (ratio drops when eyes close)
        if avg_ratio < 0.15:  # Threshold for SCRFD approximation
            self.ear_counter += 1
        else:
            if self.ear_counter >= self.EAR_CONSEC_FRAMES:
                self.blink_counter += 1
                self.has_blinked = True
                print(f"[Liveness] ✓ Blink detected (total: {self.blink_counter})")
            self.ear_counter = 0
    
    def detect_blink_mediapipe(self, landmarks):
        """Detect blinks using MediaPipe 468 landmarks (full mode)"""
        # MediaPipe eye landmarks (refined)
        left_eye = np.array([
            landmarks[33], landmarks[160], landmarks[158],
            landmarks[133], landmarks[153], landmarks[144]
        ])
        right_eye = np.array([
            landmarks[362], landmarks[385], landmarks[387],
            landmarks[263], landmarks[373], landmarks[380]
        ])
        
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < self.EAR_THRESHOLD:
            self.ear_counter += 1
        else:
            if self.ear_counter >= self.EAR_CONSEC_FRAMES:
                self.blink_counter += 1
                self.has_blinked = True
                print(f"[Liveness] ✓ Blink detected (total: {self.blink_counter}, EAR: {ear:.3f})")
            self.ear_counter = 0
    
    def compute_head_pose(self, keypoints_or_landmarks):
        """Compute simple head pose from keypoints"""
        # Extract key points
        if isinstance(keypoints_or_landmarks, np.ndarray) and keypoints_or_landmarks.shape[0] == 5:
            # SCRFD 5-point keypoints
            left_eye = keypoints_or_landmarks[0]
            right_eye = keypoints_or_landmarks[1]
            nose = keypoints_or_landmarks[2]
            
            # Compute angles
            eye_center = (left_eye + right_eye) / 2.0
            yaw = np.arctan2(nose[0] - eye_center[0], abs(nose[1] - eye_center[1]))
            pitch = np.arctan2(nose[1] - eye_center[1], np.linalg.norm(left_eye - right_eye))
            
        else:
            # MediaPipe landmarks (use key face points)
            nose_tip = keypoints_or_landmarks[1]  # Nose tip
            left_eye = keypoints_or_landmarks[33]
            right_eye = keypoints_or_landmarks[263]
            
            eye_center = (left_eye + right_eye) / 2.0
            yaw = np.arctan2(nose_tip[0] - eye_center[0], abs(nose_tip[1] - eye_center[1]))
            pitch = np.arctan2(nose_tip[1] - eye_center[1], np.linalg.norm(left_eye - right_eye))
        
        return np.degrees(yaw), np.degrees(pitch)
    
    def check_head_movement(self, current_pose):
        """Check for head pose variations (yaw, pitch changes)"""
        if self.initial_pose is None:
            self.initial_pose = current_pose
            return
        
        self.pose_history.append(current_pose)
        
        if len(self.pose_history) < 10:
            return
        
        # Calculate pose variation
        poses = np.array(self.pose_history)
        yaw_std = np.std(poses[:, 0])
        pitch_std = np.std(poses[:, 1])
        
        # Detect significant movement (> 3 degrees variation)
        if yaw_std > 3.0 or pitch_std > 3.0:
            if not self.has_head_movement:
                self.has_head_movement = True
                print(f"[Liveness] ✓ Head movement detected (yaw σ: {yaw_std:.1f}°, pitch σ: {pitch_std:.1f}°)")
    
    def compute_micro_motion(self, landmarks):
        """Compute micro-motion using landmark displacement (fool-proof method)"""
        # Select stable landmarks: cheeks, mouth corners, eyelids
        stable_indices = [234, 454, 61, 291, 33, 263]  # MediaPipe indices
        
        if self.face_mesh:  # Full MediaPipe mode
            stable_landmarks = landmarks[stable_indices]
        else:  # Fallback: use SCRFD keypoints
            stable_landmarks = landmarks  # All 5 keypoints
        
        if self.prev_landmarks is None:
            self.prev_landmarks = stable_landmarks
            return
        
        # Compute motion vectors
        motion_vectors = stable_landmarks - self.prev_landmarks
        
        # Calculate micro-motion (average magnitude)
        micro_motion = np.mean(np.linalg.norm(motion_vectors, axis=1))
        
        # Update buffer
        self.micro_motion_buffer.append(micro_motion)
        
        if len(self.micro_motion_buffer) >= self.BUFFER_SIZE:
            avg_motion = np.mean(self.micro_motion_buffer)
            
            if avg_motion > self.MICRO_MOTION_THRESH:
                if not self.has_micro_motion:
                    self.has_micro_motion = True
                    print(f"[Liveness] ✓ Micro-motion detected (avg: {avg_motion:.3f})")
        
        self.prev_landmarks = stable_landmarks
    
    def process_frame(self, frame, scrfd_keypoints=None):
        """Process frame for liveness detection"""
        if self.face_mesh and _HAS_MEDIAPIPE:
            # Full mode: Use MediaPipe for detailed analysis
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                landmarks_np = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
                
                # Blink detection
                self.detect_blink_mediapipe(landmarks_np)
                
                # Head pose tracking
                yaw, pitch = self.compute_head_pose(landmarks_np)
                self.check_head_movement((yaw, pitch))
                
                # Micro-motion analysis
                self.compute_micro_motion(landmarks_np)
        
        elif scrfd_keypoints is not None:
            # Basic mode: Use SCRFD keypoints
            self.detect_blink_scrfd(scrfd_keypoints)
            
            # Head pose
            yaw, pitch = self.compute_head_pose(scrfd_keypoints)
            self.check_head_movement((yaw, pitch))
            
            # Micro-motion (using all keypoints)
            self.compute_micro_motion(scrfd_keypoints)
    
    def is_live(self):
        """Check if person is live (not a photo/video)"""
        blink_check = self.blink_counter >= self.MIN_BLINKS
        motion_check = self.has_micro_motion
        head_check = self.has_head_movement
        
        # Require at least 2 out of 3 checks to pass
        score = sum([blink_check, motion_check, head_check])
        
        return score >= 2, {
            'blinks': self.blink_counter,
            'has_motion': motion_check,
            'has_head_movement': head_check,
            'score': score
        }
    
    def cleanup(self):
        """Release resources"""
        if self.face_mesh:
            self.face_mesh.close()


class FaceAuthenticatorBoard:
    def __init__(self, detector_path, recognizer_path, database_path):
        """Initialize face detector and recognizer for board authentication"""
        print("=" * 70)
        print("FACE AUTHENTICATION SYSTEM - IMX BOARD NPU")
        print("=" * 70)
        
        # Load SCRFD face detector (NPU)
        print(f"\n[1/3] Loading SCRFD detector (NPU)...")
        self.detector = SCRFDDetector(detector_path)
        print(f"✓ SCRFD loaded: {detector_path}")
        
        # Load Face Recognizer (NPU with Ethos-U delegate)
        print(f"\n[2/3] Loading Face Recognizer (NPU)...")
        
        # Load Ethos-U delegate for NPU acceleration
        ethosu_delegate = tflite.load_delegate(
            "/usr/lib/libethosu_delegate.so",
            {
                "device_name": "/dev/ethosu0",
                "cache_file_path": ".",
                "enable_cycle_counter": "false",
            }
        )
        print("[Auth-NPU] Ethos-U delegate loaded")
        
        self.recognizer = tflite.Interpreter(
            model_path=recognizer_path,
            experimental_delegates=[ethosu_delegate]
        )
        print("[Auth-NPU] FR model on NPU")
        
        self.recognizer.allocate_tensors()
        self.recognizer_input = self.recognizer.get_input_details()[0]
        self.recognizer_output = self.recognizer.get_output_details()[0]
        
        # Get quantization parameters
        self.fr_in_scale, self.fr_in_zero = self.recognizer_input['quantization']
        self.fr_out_scale, self.fr_out_zero = self.recognizer_output['quantization']
        
        # Get input shape
        full_shape = self.recognizer_input['shape']
        if full_shape[1] == 3:  # Channels-first
            self.input_shape = (int(full_shape[2]), int(full_shape[3]))
            self.channels_first = True
        else:  # Channels-last
            self.input_shape = (int(full_shape[1]), int(full_shape[2]))
            self.channels_first = False
        
        print(f"✓ Recognizer loaded: {recognizer_path}")
        print(f"  Input shape: {self.input_shape}, Channels-first: {self.channels_first}")
        print(f"  FR Input: scale={self.fr_in_scale}, zero={self.fr_in_zero}")
        print(f"  FR Output: scale={self.fr_out_scale}, zero={self.fr_out_zero}")
        
        # Load enrolled profiles
        print(f"\n[3/3] Loading enrolled profiles...")
        self.enrolled_profiles = self.load_database(database_path)
        
        if len(self.enrolled_profiles) == 0:
            print("⚠ WARNING: No enrolled profiles found!")
            print("  Please run: python3 enroll_industrial_board.py")
        else:
            print(f"✓ Loaded {len(self.enrolled_profiles)} enrolled driver(s):")
            for name, profile in self.enrolled_profiles.items():
                driver_id = profile.get('driver_id', 'N/A')
                print(f"    • {name} (ID: {driver_id})")
        
        print("=" * 70)
        
        # Authentication parameters
        self.similarity_threshold = 0.65      # Main threshold for match
        self.strict_threshold = 0.75          # High confidence threshold
        self.consecutive_frames = 3           # Frames needed for stable recognition
        
        # Multi-frame averaging
        self.embedding_history = deque(maxlen=5)
        
        # State tracking
        self.current_driver = None
        self.match_streak = 0
        self.no_face_frames = 0
        self.no_face_threshold = 10
        
        # Unknown tracking
        self.unknown_streak = 0
        self.unknown_alert_threshold = 30  # 1 second at 30fps
        self.unknown_logged = False
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()

    def load_database(self, path):
        """Load enrolled profiles from database"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            profiles = {}
            for profile in data.get('profiles', []):
                name = profile['name']
                embedding = np.array(profile['embedding'], dtype=np.float32)
                # L2 normalize
                embedding = embedding / np.linalg.norm(embedding)
                
                profiles[name] = {
                    'embedding': embedding,
                    'driver_id': profile.get('driver_id', len(profiles) + 1),
                    'enrollment_date': profile.get('enrollment_date', 'Unknown')
                }
            
            return profiles
        except FileNotFoundError:
            print(f"✗ Database not found: {path}")
            return {}
        except Exception as e:
            print(f"✗ Error loading database: {e}")
            return {}

    def apply_lighting_normalization(self, face):
        """4-stage lighting normalization pipeline"""
        # Stage 1: Adaptive Gamma Correction
        mean_val = np.mean(face)
        if mean_val < 80:
            gamma = 1.5
        elif mean_val > 170:
            gamma = 0.7
        else:
            gamma = 1.0
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            face = cv2.LUT(face, table)
        
        # Stage 2: Multi-scale Retinex (MSR)
        face_float = face.astype(np.float32) + 1.0
        scales = [15, 80, 250]
        retinex = np.zeros_like(face_float)
        
        for scale in scales:
            blurred = cv2.GaussianBlur(face_float, (0, 0), scale)
            retinex += np.log10(face_float) - np.log10(blurred + 1.0)
        
        retinex = retinex / len(scales)
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
        face = (retinex * 255).astype(np.uint8)
        
        # Stage 3: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(face.shape) == 3:
            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            face = clahe.apply(face)
        
        # Stage 4: Fixed Standardization
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        
        return face

    def extract_embedding(self, aligned_face):
        """Extract face embedding with lighting normalization"""
        # Apply preprocessing
        face_normalized = self.apply_lighting_normalization(aligned_face)
        
        # Ensure correct shape
        if len(face_normalized.shape) == 2:
            face_normalized = cv2.cvtColor((face_normalized * 128 + 127.5).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            face_normalized = (face_normalized - 127.5) / 128.0
        
        # Quantize for INT8 model
        if self.fr_in_scale > 0:
            face_quantized = (face_normalized / self.fr_in_scale) + self.fr_in_zero
            face_quantized = np.clip(face_quantized, -128, 127).astype(np.int8)
        else:
            face_quantized = face_normalized.astype(np.float32)
        
        face_input = np.expand_dims(face_quantized, axis=0)
        
        # Handle channels-first format if needed
        if self.channels_first and len(face_input.shape) == 4:
            face_input = np.transpose(face_input, (0, 3, 1, 2))
        
        # Run inference on NPU
        self.recognizer.set_tensor(self.recognizer_input['index'], face_input)
        self.recognizer.invoke()
        
        # Get embedding
        embedding_raw = self.recognizer.get_tensor(self.recognizer_output['index'])
        
        # Dequantize
        if self.fr_out_scale > 0:
            embedding = (embedding_raw.astype(np.float32) - self.fr_out_zero) * self.fr_out_scale
        else:
            embedding = embedding_raw.astype(np.float32)
        
        # Normalize
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

    def cleanup(self):
        """Release all resources and free memory"""
        try:
            # Close detector
            if hasattr(self, 'detector') and self.detector:
                if hasattr(self.detector, 'interpreter'):
                    del self.detector.interpreter
                del self.detector
            
            # Close recognizer
            if hasattr(self, 'recognizer') and self.recognizer:
                del self.recognizer
            
            # Clear profiles
            if hasattr(self, 'enrolled_profiles'):
                self.enrolled_profiles.clear()
            
            print("[Auth-Cleanup] Face authentication models unloaded")
        except Exception as e:
            print(f"[Auth-Cleanup] Cleanup error: {e}")
    
    def authenticate(self, embedding):
        """Authenticate face against enrolled profiles"""
        if len(self.enrolled_profiles) == 0:
            return None, 0.0, False
        
        # Add to history for averaging
        self.embedding_history.append(embedding)
        
        # Use averaged embedding if we have enough samples
        if len(self.embedding_history) >= 3:
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
        
        # Get best match
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_name, best_sim = sorted_matches[0]
        
        # Check threshold
        if best_sim < self.similarity_threshold:
            return None, best_sim, False
        
        # Check consecutive frames for stability
        if self.current_driver == best_name:
            self.match_streak += 1
        else:
            self.current_driver = best_name
            self.match_streak = 1
        
        # Require consecutive frames
        is_authenticated = self.match_streak >= self.consecutive_frames
        
        return best_name, best_sim, is_authenticated

    def run(self, camera_id=0):
        """Run live authentication on board"""
        print("\n" + "=" * 70)
        print("STARTING AUTHENTICATION ON IMX BOARD")
        print("=" * 70)
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset authentication state")
        print("=" * 70 + "\n")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("✗ Cannot open camera!")
            return
        
        authenticated_driver = None
        authenticated_similarity = 0.0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Failed to read frame")
                break
            
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
            self.last_time = current_time
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
            
            # Detect faces using NPU
            detections = self.detector.detect(frame, conf_threshold=0.60)
            
            if len(detections) == 0:
                # No face detected
                self.no_face_frames += 1
                
                # Reset if no face for threshold frames
                if self.no_face_frames >= self.no_face_threshold:
                    if authenticated_driver:
                        print(f"[{frame_count}] Driver {authenticated_driver} left (no face)")
                    authenticated_driver = None
                    authenticated_similarity = 0.0
                    self.current_driver = None
                    self.match_streak = 0
                    self.embedding_history.clear()
                    self.unknown_streak = 0
                    self.unknown_logged = False
                
                # Display
                cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                if authenticated_driver:
                    cv2.putText(frame, f"Last: {authenticated_driver}", (20, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Face detected
                self.no_face_frames = 0
                
                det = detections[0]  # Use best detection
                box = det['box']
                keypoints = det['landmarks']
                score = det['score']
                
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # If already authenticated, just show result
                if authenticated_driver:
                    name = authenticated_driver
                    similarity = authenticated_similarity
                    color = (0, 255, 0)  # Green
                    status = "AUTHENTICATED"
                    self.unknown_streak = 0
                    self.unknown_logged = False
                else:
                    # Run authentication
                    aligned_face = align_face(frame, keypoints, target_size=self.input_shape[0])
                    
                    # Validate aligned face
                    if aligned_face is None or aligned_face.size == 0:
                        color = (0, 0, 255)  # Red
                        status = "ALIGNMENT FAILED"
                        name = "Error"
                    else:
                        embedding = self.extract_embedding(aligned_face)
                        name, similarity, is_authenticated = self.authenticate(embedding)
                        
                        if is_authenticated:
                            authenticated_driver = name
                            authenticated_similarity = similarity
                            color = (0, 255, 0)  # Green
                            status = "AUTHENTICATED"
                            driver_id = self.enrolled_profiles[name]['driver_id']
                            print(f"[{frame_count}] ✓ AUTHENTICATED: {name} (ID: {driver_id}) | Similarity: {similarity:.3f}")
                            self.unknown_streak = 0
                            self.unknown_logged = False
                        elif name:
                            color = (0, 165, 255)  # Orange
                            status = f"VERIFYING... ({self.match_streak}/{self.consecutive_frames})"
                            self.unknown_streak = 0
                        else:
                            color = (0, 0, 255)  # Red
                            status = "UNKNOWN"
                            name = "Unknown"
                            
                            # Track unknown streak
                            self.unknown_streak += 1
                            
                            # Alert after sustained unknown presence
                            if self.unknown_streak >= self.unknown_alert_threshold:
                                if not self.unknown_logged:
                                    print(f"[{frame_count}] ⚠ ALERT: UNKNOWN PERSON DETECTED!")
                                    print(f"  Similarity to nearest: {similarity:.1%}")
                                    print(f"  Duration: {self.unknown_streak} frames")
                                    # TODO: Add buzzer trigger here
                                    # trigger_buzzer("UNAUTHORIZED_ACCESS")
                                    self.unknown_logged = True
                                
                                # Show warning overlay
                                cv2.putText(frame, "! UNAUTHORIZED PERSON !", 
                                           (20, frame.shape[0] - 50),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw 5-point landmarks
                kps = keypoints.reshape(5, 2).astype(int)
                for kx, ky in kps:
                    cv2.circle(frame, (kx, ky), 3, (0, 255, 255), -1)
                
                # Display driver info
                y_offset = y1 - 15
                
                # Name
                cv2.putText(frame, name, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Status
                y_offset -= 30
                cv2.putText(frame, status, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Similarity (only if authenticated or verifying)
                if name not in ["Unknown", "Error"]:
                    y_offset -= 25
                    cv2.putText(frame, f"Match: {similarity:.1%}", (x1, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display enrolled count
            cv2.putText(frame, f"Enrolled: {len(self.enrolled_profiles)}", (frame.shape[1] - 150, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Face Authentication - IMX Board', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print(f"[{frame_count}] Manual reset")
                authenticated_driver = None
                authenticated_similarity = 0.0
                self.current_driver = None
                self.match_streak = 0
                self.embedding_history.clear()
                self.no_face_frames = 0
                self.unknown_streak = 0
                self.unknown_logged = False
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("AUTHENTICATION STOPPED")
        print("=" * 70)


def quick_authenticate(detector_path="scrfd_500m_full_int8_vela.tflite",
                       recognizer_path="fr_int8_velaS.tflite",
                       database_path="drivers.json",
                       camera_id=0,
                       timeout_seconds=10,
                       enable_liveness=True,
                       liveness_blink_min=2,
                       liveness_motion_thresh=0.8):
    """
    Quick authentication for DMS integration with anti-spoofing liveness detection.
    
    Returns: (authenticated, driver_name, driver_id)
    - authenticated: True if driver recognized AND passed liveness, False otherwise
    - driver_name: Name of driver if authenticated, None otherwise
    - driver_id: Driver ID if authenticated, None otherwise
    
    Liveness Detection:
    - Prevents photo/video spoofing attacks
    - Checks for eye blinks, head movement, and micro-motion
    - Requires 2/3 liveness checks to pass
    """
    print("\n" + "=" * 70)
    print("DMS DRIVER AUTHENTICATION + ANTI-SPOOFING")
    print("=" * 70)
    
    authenticator = FaceAuthenticatorBoard(detector_path, recognizer_path, database_path)
    
    # Initialize liveness detector
    liveness_detector = None
    if enable_liveness:
        liveness_detector = LivenessDetector(
            blink_threshold=liveness_blink_min,
            motion_threshold=liveness_motion_thresh,
            buffer_size=15
        )
        print(f"[Liveness] Anti-spoofing ENABLED (timeout: {timeout_seconds}s)")
    else:
        print("[Liveness] Anti-spoofing DISABLED")
    
    if len(authenticator.enrolled_profiles) == 0:
        print("⚠ No enrolled drivers - skipping authentication")
        return False, None, None
    
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("✗ Cannot open camera - skipping authentication")
        return False, None, None
    
    start_time = time.time()
    authenticated_driver = None
    authenticated_id = None
    frame_count = 0
    liveness_passed = False
    
    print(f"\nAuthenticating driver... (timeout: {timeout_seconds}s)")
    if enable_liveness:
        print("Anti-spoofing active: Checking for blinks, head movement, and micro-motion")
    print("Look at the camera for identification")
    print("=" * 70 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        # Check timeout
        if elapsed >= timeout_seconds:
            if enable_liveness and not liveness_passed:
                print(f"[Authentication] \u2717 FAILED - Liveness check timeout")
                print(f"[Authentication]   Possible photo/video spoofing detected")
                print(f"[Authentication]   Blinks: {liveness_detector.blink_counter}/{liveness_detector.MIN_BLINKS}")
                print(f"[Authentication]   Micro-motion: {'PASS' if liveness_detector.has_micro_motion else 'FAIL'}")
                print(f"[Authentication]   Head movement: {'PASS' if liveness_detector.has_head_movement else 'FAIL'}\n")
            else:
                print(f"[Authentication] Timeout - No authorized driver in {timeout_seconds}s")
                print(f"[Authentication] Proceeding with monitoring in unauthorized mode\n")
            
            cap.release()
            if liveness_detector:
                liveness_detector.cleanup()
            authenticator.cleanup()
            del authenticator
            return False, None, None
        
        # Start latency tracking for this frame
        if _LAT: lat.start_frame(frame_count)
        
        # Camera preprocessing (already read, just RGB conversion for tracking)
        if _LAT: lat.mark('t_cap', frame_count)
        
        # Detect faces with SCRFD
        detections = authenticator.detector.detect(frame, conf_threshold=0.60)
        if _LAT: lat.mark('t_scrfd', frame_count)
        
        if len(detections) == 0:
            # No face - print every 2 seconds
            if frame_count % 60 == 0:
                remaining = int(timeout_seconds - elapsed)
                print(f"[Authentication] No face detected - timeout in {remaining}s")
        else:
            det = detections[0]
            keypoints = det['landmarks']
            
            # ===== LIVENESS DETECTION =====
            if enable_liveness and liveness_detector:
                liveness_detector.process_frame(frame, scrfd_keypoints=keypoints)
                
                is_live, liveness_stats = liveness_detector.is_live()
                
                if is_live and not liveness_passed:
                    liveness_passed = True
                    print(f"[Liveness] \u2713 PASSED - Real person detected")
                    print(f"[Liveness]   Blinks: {liveness_stats['blinks']}")
                    print(f"[Liveness]   Micro-motion: {'PASS' if liveness_stats['has_motion'] else 'FAIL'}")
                    print(f"[Liveness]   Head movement: {'PASS' if liveness_stats['has_head_movement'] else 'FAIL'}")
                    print(f"[Liveness]   Score: {liveness_stats['score']}/3\n")
            
            # ===== FACE RECOGNITION (only if liveness passed or disabled) =====
            if not enable_liveness or liveness_passed:
                aligned_face = align_face(frame, keypoints, target_size=authenticator.input_shape[0])
                
                if aligned_face is not None and aligned_face.size > 0:
                    embedding = authenticator.extract_embedding(aligned_face)
                    if _LAT: lat.mark('t_facerec', frame_count)  # Mark after MobileFaceNet inference
                    
                    name, similarity, is_authenticated = authenticator.authenticate(embedding)
                    
                    if is_authenticated:
                        authenticated_driver = name
                        authenticated_id = authenticator.enrolled_profiles[name]['driver_id']
                        
                        # Log face recognition latency
                        if _LAT:
                            lat.log_facerec_pipeline(frame_count)
                        
                        print(f"[Authentication] \u2713 AUTHENTICATED: {name} (ID: {authenticated_id})")
                        print(f"[Authentication]   Similarity: {similarity:.1%}")
                        print(f"[Authentication]   Time: {elapsed:.1f}s")
                        if enable_liveness:
                            print(f"[Authentication]   Liveness: VERIFIED")
                        print()
                        
                        cap.release()
                        if liveness_detector:
                            liveness_detector.cleanup()
                        authenticator.cleanup()
                        del authenticator
                        return True, authenticated_driver, authenticated_id
                    
                    elif name:
                        # Verifying - print every 30 frames
                        if frame_count % 30 == 0:
                            print(f"[Authentication] Verifying {name}... ({authenticator.match_streak}/{authenticator.consecutive_frames})")
                    else:
                        # Unknown - print every 30 frames
                        if frame_count % 30 == 0:
                            remaining = int(timeout_seconds - elapsed)
                            print(f"[Authentication] Unknown person - timeout in {remaining}s")
            else:
                # Still checking liveness
                if frame_count % 60 == 0:
                    remaining = int(timeout_seconds - elapsed)
                    print(f"[Liveness] Checking for signs of life... ({remaining}s remaining)")
    
    cap.release()
    if liveness_detector:
        liveness_detector.cleanup()
    authenticator.cleanup()
    del authenticator
    return False, None, None


if __name__ == "__main__":
    # Standalone mode - continuous authentication
    authenticator = FaceAuthenticatorBoard(
        detector_path="scrfd_500m_full_int8_vela.tflite",
        recognizer_path="fr_int8_velaS.tflite",
        database_path="drivers.json"
    )
    authenticator.run(camera_id=0)
