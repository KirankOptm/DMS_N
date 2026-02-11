"""
Face Authentication System - Live Camera
Authenticates enrolled drivers using SCRFD detection + Face Recognition
Uses same models as enroll_industrial.py
"""

import cv2
import numpy as np
import json
import time
from collections import deque
import tensorflow as tf
from scrfd_detector import SCRFDDetector, align_face


class FaceAuthenticator:
    def __init__(self, detector_path, recognizer_path, database_path):
        """Initialize face detector and recognizer for authentication"""
        print("=" * 70)
        print("FACE AUTHENTICATION SYSTEM")
        print("=" * 70)
        
        # Load SCRFD face detector
        print(f"\n[1/3] Loading SCRFD detector...")
        self.detector = SCRFDDetector(detector_path)
        print(f"✓ SCRFD loaded: {detector_path}")
        
        # Load Face Recognizer
        print(f"\n[2/3] Loading Face Recognizer...")
        self.recognizer = tf.lite.Interpreter(model_path=recognizer_path)
        self.recognizer.allocate_tensors()
        self.recognizer_input = self.recognizer.get_input_details()[0]
        self.recognizer_output = self.recognizer.get_output_details()[0]
        
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
        
        # Load enrolled profiles
        print(f"\n[3/3] Loading enrolled profiles...")
        self.enrolled_profiles = self.load_database(database_path)
        
        if len(self.enrolled_profiles) == 0:
            print("⚠ WARNING: No enrolled profiles found!")
            print("  Please run: python enroll_industrial.py")
        else:
            print(f"✓ Loaded {len(self.enrolled_profiles)} enrolled driver(s):")
            for name, profile in self.enrolled_profiles.items():
                driver_id = profile.get('driver_id', 'N/A')
                print(f"    • {name} (ID: {driver_id})")
        
        print("=" * 70)
        
        # Authentication parameters
        self.similarity_threshold = 0.75      # Main threshold for match
        self.strict_threshold = 0.85          # High confidence threshold
        self.consecutive_frames = 3           # Frames needed for stable recognition
        
        # Multi-frame averaging
        self.embedding_history = deque(maxlen=5)
        
        # State tracking
        self.current_driver = None
        self.match_streak = 0
        self.no_face_frames = 0
        self.no_face_threshold = 10
        
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

    def extract_embedding(self, aligned_face):
        """Extract face embedding with lighting normalization"""
        # STAGE 1: Gamma correction
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 80:
            gamma = 1.5
        elif mean_brightness > 175:
            gamma = 0.7
        else:
            gamma = 1.0
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            aligned_face = cv2.LUT(aligned_face, table)
        
        # STAGE 2: Multi-scale Retinex
        def single_scale_retinex(img, sigma):
            retinex = np.log10(img.astype(np.float32) + 1.0) - np.log10(cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma) + 1.0)
            return retinex
        
        img_msr = np.zeros_like(aligned_face, dtype=np.float32)
        for i in range(3):
            msr_channel = (single_scale_retinex(aligned_face[:,:,i], 15) + 
                          single_scale_retinex(aligned_face[:,:,i], 80) + 
                          single_scale_retinex(aligned_face[:,:,i], 200)) / 3.0
            img_msr[:,:,i] = msr_channel
        
        img_msr = (img_msr - np.min(img_msr)) / (np.max(img_msr) - np.min(img_msr) + 1e-6) * 255.0
        img_msr = np.clip(img_msr, 0, 255).astype(np.uint8)
        
        # STAGE 3: CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_eq = np.zeros_like(img_msr, dtype=np.uint8)
        for i in range(3):
            img_eq[:,:,i] = clahe.apply(img_msr[:,:,i])
        
        # STAGE 4: Normalize and quantize
        img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_normalized = (img_rgb - 127.5) / 128.0
        
        scale = self.recognizer_input['quantization'][0]
        zero_point = self.recognizer_input['quantization'][1]
        img_int8 = (img_normalized / scale + zero_point).astype(np.int8)
        
        input_data = np.expand_dims(img_int8, axis=0)
        if self.channels_first:
            input_data = np.transpose(input_data, (0, 3, 1, 2))
        
        # Run inference
        self.recognizer.set_tensor(self.recognizer_input['index'], input_data)
        self.recognizer.invoke()
        embedding = self.recognizer.get_tensor(self.recognizer_output['index'])[0]
        
        # Dequantize
        out_scale, out_zero = self.recognizer_output['quantization']
        embedding = (embedding.astype(np.float32) - out_zero) * out_scale
        
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

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
        """Run live authentication"""
        print("\n" + "=" * 70)
        print("STARTING AUTHENTICATION")
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
            
            # Detect faces
            detections = self.detector.detect(frame, score_threshold=0.6)
            
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
                keypoints = det['keypoints']
                score = det['score']
                
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # If already authenticated, just show result
                if authenticated_driver:
                    name = authenticated_driver
                    similarity = authenticated_similarity
                    color = (0, 255, 0)  # Green
                    status = "AUTHENTICATED"
                else:
                    # Run authentication
                    aligned_face, _ = align_face(frame, keypoints, target_size=self.input_shape[0])
                    
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
                        elif name:
                            color = (0, 165, 255)  # Orange
                            status = f"VERIFYING... ({self.match_streak}/{self.consecutive_frames})"
                        else:
                            color = (0, 0, 255)  # Red
                            status = "UNKNOWN"
                            name = "Unknown"
                
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
                if name != "Unknown":
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
            cv2.imshow('Face Authentication', frame)
            
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
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("AUTHENTICATION STOPPED")
        print("=" * 70)


if __name__ == "__main__":
    authenticator = FaceAuthenticator(
        detector_path="scrfd_500m_full_int8.tflite",
        recognizer_path="fr_int8.tflite",
        database_path="drivers.json"
    )
    authenticator.run(camera_id=0)
