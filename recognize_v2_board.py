"""
Face Recognition Module - IMX Board NPU Version
Uses vela-optimized FR model on NPU, standard landmark model on CPU
"""

import numpy as np
import cv2
import json
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("[WARNING] tflite_runtime not found, falling back to tensorflow.lite")
    import tensorflow.lite as tflite

from scrfd_detector_board import SCRFDDetector, align_face


class FaceRecognizer:
    def __init__(self, 
                 detector_path="scrfd_500m_full_int8_vela.tflite",
                 recognizer_path="fr_int8_velaS.tflite",
                 database_path="drivers.json"):
        
        print("[FaceRec-NPU] Initializing recognition system...")
        
        # Load detector
        self.detector = SCRFDDetector(detector_path)
        
        # Load face recognition model (NPU)
        print(f"[FaceRec-NPU] Loading FR model: {recognizer_path}")
        ethosu_delegate = tflite.load_delegate(
            "/usr/lib/libethosu_delegate.so",
            {
                "device_name": "/dev/ethosu0",
                "cache_file_path": ".",
                "enable_cycle_counter": "false",
            }
        )
        print("[FaceRec-NPU] Ethos-U delegate loaded")
        
        self.fr_interpreter = tflite.Interpreter(
            model_path=recognizer_path,
            experimental_delegates=[ethosu_delegate]
        )
        print("[FaceRec-NPU] FR model on NPU")
        
        self.fr_interpreter.allocate_tensors()
        
        self.fr_input_details = self.fr_interpreter.get_input_details()[0]
        self.fr_output_details = self.fr_interpreter.get_output_details()[0]
        
        self.fr_in_scale, self.fr_in_zero = self.fr_input_details['quantization']
        self.fr_out_scale, self.fr_out_zero = self.fr_output_details['quantization']
        
        print(f"[FaceRec-NPU] FR Input: scale={self.fr_in_scale}, zero={self.fr_in_zero}")
        print(f"[FaceRec-NPU] FR Output: scale={self.fr_out_scale}, zero={self.fr_out_zero}")
        
        # Load database
        self.database_path = database_path
        self.load_database()
        
        # Recognition params
        self.similarity_threshold = 0.45
        self.strict_threshold = 0.50
        self.margin = 0.10
        self.consecutive_frames = 2
        self.variance_threshold = 0.12
        self.min_similarity = 0.45
        
        # State tracking
        self.recognition_buffer = []
        self.buffer_size = 5
        
        print("[FaceRec-NPU] Initialization complete")
    
    def load_database(self):
        """Load enrolled drivers database"""
        if not os.path.exists(self.database_path):
            print(f"[FaceRec-NPU] Database not found: {self.database_path}")
            self.drivers = []
            return
        
        with open(self.database_path, 'r') as f:
            data = json.load(f)
            self.drivers = data.get('profiles', [])
        
        print(f"[FaceRec-NPU] Loaded {len(self.drivers)} drivers from database")
    
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
        
        # Run inference
        self.fr_interpreter.set_tensor(self.fr_input_details['index'], face_input)
        self.fr_interpreter.invoke()
        
        # Get embedding
        embedding_raw = self.fr_interpreter.get_tensor(self.fr_output_details['index'])
        
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
    
    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6)
    
    def recognize_face(self, frame):
        """Recognize face in frame with multi-stage validation"""
        # Detect faces
        detections = self.detector.detect(frame, conf_threshold=0.50)
        
        if not detections:
            return None
        
        # Get largest face
        detection = max(detections, key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]))
        
        # Align face
        aligned = align_face(frame, detection['landmarks'], target_size=112)
        
        # Extract embedding
        embedding = self.extract_embedding(aligned)
        
        # Match against database
        best_match = None
        best_similarity = 0.0
        
        for driver in self.drivers:
            db_embedding = np.array(driver['embedding'])
            similarity = self.cosine_similarity(embedding, db_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = driver
        
        # Multi-stage validation
        if best_match is None or best_similarity < self.similarity_threshold:
            return {
                'driver_id': None,
                'name': 'Unknown',
                'similarity': best_similarity,
                'detection': detection,
                'aligned_face': aligned
            }
        
        # Check margin (2nd best must be sufficiently worse)
        second_best = 0.0
        for driver in self.drivers:
            if driver['driver_id'] == best_match['driver_id']:
                continue
            db_embedding = np.array(driver['embedding'])
            similarity = self.cosine_similarity(embedding, db_embedding)
            second_best = max(second_best, similarity)
        
        margin = best_similarity - second_best
        if margin < self.margin:
            return {
                'driver_id': None,
                'name': 'Unknown',
                'similarity': best_similarity,
                'detection': detection,
                'aligned_face': aligned
            }
        
        # Temporal consistency check
        self.recognition_buffer.append({
            'driver_id': best_match['driver_id'],
            'similarity': best_similarity
        })
        
        if len(self.recognition_buffer) > self.buffer_size:
            self.recognition_buffer.pop(0)
        
        if len(self.recognition_buffer) >= self.consecutive_frames:
            recent = self.recognition_buffer[-self.consecutive_frames:]
            ids = [r['driver_id'] for r in recent]
            sims = [r['similarity'] for r in recent]
            
            # Check consistency
            if len(set(ids)) == 1 and np.std(sims) < self.variance_threshold:
                return {
                    'driver_id': best_match['driver_id'],
                    'name': best_match['name'],
                    'similarity': best_similarity,
                    'detection': detection,
                    'aligned_face': aligned
                }
        
        # Not consistent yet
        return {
            'driver_id': None,
            'name': 'Verifying...',
            'similarity': best_similarity,
            'detection': detection,
            'aligned_face': aligned
        }
