"""
Industrial Face Enrollment System - IMX Board NPU Version
Uses vela-optimized models on NPU
"""

import cv2
import numpy as np
import json
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from scrfd_detector_board import SCRFDDetector, align_face


class FaceEnroller:
    def __init__(self, detector_path="scrfd_500m_full_int8_vela.tflite", 
                 recognizer_path="fr_int8_velaS.tflite"):
        """Initialize face detector and recognizer with NPU"""
        print("[Enroll-NPU] Initializing enrollment system...")
        
        # Load SCRFD detector
        self.detector = SCRFDDetector(detector_path)
        
        # Load face recognition model (NPU)
        print(f"[Enroll-NPU] Loading FR model: {recognizer_path}")
        ethosu_delegate = tflite.load_delegate(
            "libethosu_delegate.so",
            {
                "device_name": "ethos-u0",
                "cache_file_path": ".",
                "enable_cycle_counter": "false",
            }
        )
        print("[Enroll-NPU] Ethos-U delegate loaded")
        
        self.recognizer = tflite.Interpreter(
            model_path=recognizer_path,
            experimental_delegates=[ethosu_delegate]
        )
        print("[Enroll-NPU] FR model on NPU")
        
        self.recognizer.allocate_tensors()
        self.recognizer_input = self.recognizer.get_input_details()[0]
        self.recognizer_output = self.recognizer.get_output_details()[0]
        
        # Quantization params
        self.in_scale, self.in_zero = self.recognizer_input['quantization']
        self.out_scale, self.out_zero = self.recognizer_output['quantization']
        
        print(f"[Enroll-NPU] FR Input: scale={self.in_scale}, zero={self.in_zero}")
        print(f"[Enroll-NPU] FR Output: scale={self.out_scale}, zero={self.out_zero}")
    
    def detect_faces(self, frame):
        """Detect faces using SCRFD"""
        detections = self.detector.detect(frame, conf_threshold=0.75)
        return detections
    
    def apply_lighting_normalization(self, face):
        """4-stage lighting normalization (same as recognize_v2_board.py)"""
        # Stage 1: Adaptive Gamma
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
        
        # Stage 2: Multi-scale Retinex
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
        """Extract face embedding with preprocessing"""
        # Apply lighting normalization
        face_normalized = self.apply_lighting_normalization(aligned_face)
        
        # Ensure RGB
        if len(face_normalized.shape) == 2:
            face_normalized = cv2.cvtColor((face_normalized * 128 + 127.5).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            face_normalized = (face_normalized - 127.5) / 128.0
        
        # Quantize for INT8
        if self.in_scale > 0:
            face_quantized = (face_normalized / self.in_scale) + self.in_zero
            face_quantized = np.clip(face_quantized, -128, 127).astype(np.int8)
        else:
            face_quantized = face_normalized.astype(np.float32)
        
        face_input = np.expand_dims(face_quantized, axis=0)
        
        # Run inference
        self.recognizer.set_tensor(self.recognizer_input['index'], face_input)
        self.recognizer.invoke()
        
        # Get embedding
        embedding_raw = self.recognizer.get_tensor(self.recognizer_output['index'])
        
        # Dequantize
        if self.out_scale > 0:
            embedding = (embedding_raw.astype(np.float32) - self.out_zero) * self.out_scale
        else:
            embedding = embedding_raw.astype(np.float32)
        
        # Normalize
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def apply_lighting_augmentation(self, face):
        """Apply 6 lighting augmentations"""
        augmented = []
        
        # 1. Original
        augmented.append(face.copy())
        
        # 2. Darker (gamma = 1.8)
        dark = np.clip((face / 255.0) ** 1.8 * 255, 0, 255).astype(np.uint8)
        augmented.append(dark)
        
        # 3. Brighter (gamma = 0.6)
        bright = np.clip((face / 255.0) ** 0.6 * 255, 0, 255).astype(np.uint8)
        augmented.append(bright)
        
        # 4. High contrast (alpha = 1.3)
        high_contrast = np.clip(face * 1.3, 0, 255).astype(np.uint8)
        augmented.append(high_contrast)
        
        # 5. Low contrast (alpha = 0.7)
        low_contrast = np.clip(face * 0.7, 0, 255).astype(np.uint8)
        augmented.append(low_contrast)
        
        # 6. Shadow simulation
        h, w = face.shape[:2]
        shadow = face.copy().astype(np.float32)
        gradient = np.linspace(1.0, 0.5, h).reshape(-1, 1)
        shadow = (shadow * gradient).astype(np.uint8)
        augmented.append(shadow)
        
        return augmented
    
    def enroll_driver(self, driver_id, name, database_path="drivers.json"):
        """Multi-pose enrollment with augmentation - Headless mode for IMX board"""
        print("\n" + "="*70)
        print("DRIVER ENROLLMENT - HEADLESS MODE (IMX BOARD)")
        print("="*70)
        print(f"Driver ID: {driver_id}")
        print(f"Name: {name}")
        print(f"Camera: 40-45 degree angle supported")
        print("="*70)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return False
        
        # Multi-pose capture
        poses = [
            ("FRONTAL", 10),
            ("LEFT", 10),
            ("RIGHT", 10)
        ]
        
        all_embeddings = []
        pose_idx = 0
        current_pose, frames_needed = poses[pose_idx]
        frames_captured = 0
        no_face_count = 0
        
        print(f"\n[POSE] {current_pose} - Capture {frames_needed} frames")
        print("[INFO] Camera angle: Works with 40-45 degree mounting")
        print("[INFO] Look at camera and hold position...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Detect faces
            detections = self.detect_faces(frame)
            
            if detections:
                no_face_count = 0
                det = detections[0]
                box = det['box']
                landmarks = det['landmarks']
                
                # Capture frame
                if frames_captured < frames_needed:
                    aligned = align_face(frame, landmarks, target_size=112)
                    
                    if aligned is not None and aligned.size > 0:
                        # Apply augmentation
                        augmented_faces = self.apply_lighting_augmentation(aligned)
                        
                        for aug_face in augmented_faces:
                            emb = self.extract_embedding(aug_face)
                            all_embeddings.append(emb)
                        
                        frames_captured += 1
                        print(f"  [{frames_captured}/{frames_needed}] Captured (6 augmentations)")
                        
                        if frames_captured >= frames_needed:
                            pose_idx += 1
                            if pose_idx < len(poses):
                                current_pose, frames_needed = poses[pose_idx]
                                frames_captured = 0
                                print(f"\n[POSE] {current_pose} - Capture {frames_needed} frames")
                            else:
                                print("\n[COMPLETE] All poses captured!")
                                break
                    else:
                        print("  [WARNING] Face alignment failed, retrying...")
            else:
                no_face_count += 1
                if no_face_count % 30 == 0:  # Print every second
                    print(f"  [WAITING] No face detected - please look at camera")
                
                # Timeout after 30 seconds of no face
                if no_face_count > 900:  # 30 seconds at 30fps
                    print("\n[TIMEOUT] No face detected for 30 seconds")
                    cap.release()
                    return False
        
        cap.release()
        
        if len(all_embeddings) == 0:
            print("[ERROR] No embeddings captured")
            return False
        
        # Production-level averaging
        print(f"\n[PROCESSING] Total embeddings: {len(all_embeddings)}")
        
        embeddings_array = np.array(all_embeddings)
        
        # Remove outliers (bottom 20%)
        similarities = []
        mean_emb = np.mean(embeddings_array, axis=0)
        for emb in embeddings_array:
            sim = np.dot(emb, mean_emb) / (np.linalg.norm(emb) * np.linalg.norm(mean_emb) + 1e-6)
            similarities.append(sim)
        
        threshold = np.percentile(similarities, 20)
        filtered = embeddings_array[np.array(similarities) >= threshold]
        
        print(f"[PROCESSING] After outlier removal: {len(filtered)} embeddings")
        
        # Weighted averaging
        weights = []
        mean_emb = np.mean(filtered, axis=0)
        for emb in filtered:
            sim = np.dot(emb, mean_emb) / (np.linalg.norm(emb) * np.linalg.norm(mean_emb) + 1e-6)
            weights.append(sim)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        final_embedding = np.average(filtered, axis=0, weights=weights)
        final_embedding = final_embedding / (np.linalg.norm(final_embedding) + 1e-6)
        
        # Quality metrics
        sims = [np.dot(final_embedding, emb) / (np.linalg.norm(final_embedding) * np.linalg.norm(emb) + 1e-6) 
                for emb in filtered]
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        min_sim = np.min(sims)
        
        print(f"\n[QUALITY METRICS]")
        print(f"  Mean Similarity: {mean_sim:.3f}")
        print(f"  Std Deviation: {std_sim:.3f}")
        print(f"  Min Similarity: {min_sim:.3f}")
        
        # Validate quality
        if mean_sim < 0.78 or std_sim > 0.12 or min_sim < 0.65:
            print("[WARNING] Low quality enrollment! Consider re-enrolling.")
        
        # Check for duplicates
        if Path(database_path).exists():
            with open(database_path, 'r') as f:
                data = json.load(f)
                profiles = data.get('profiles', [])
            
            for profile in profiles:
                db_emb = np.array(profile['embedding'])
                sim = np.dot(final_embedding, db_emb) / (np.linalg.norm(final_embedding) * np.linalg.norm(db_emb) + 1e-6)
                
                if sim > 0.85:
                    print(f"\n[WARNING] High similarity ({sim:.3f}) with existing driver: {profile['name']}")
                    print("This might be a duplicate enrollment!")
                    response = input("Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        return False
                elif sim > 0.75:
                    print(f"\n[INFO] Moderate similarity ({sim:.3f}) with {profile['name']}")
        
        # Save to database
        profile = {
            'driver_id': driver_id,
            'name': name,
            'embedding': final_embedding.tolist(),
            'enrollment_date': str(np.datetime64('now')),
            'quality_metrics': {
                'mean_similarity': float(mean_sim),
                'std_deviation': float(std_sim),
                'min_similarity': float(min_sim),
                'num_embeddings': int(len(filtered))
            }
        }
        
        if Path(database_path).exists():
            with open(database_path, 'r') as f:
                data = json.load(f)
        else:
            data = {'profiles': []}
        
        # Remove old profile with same driver_id
        data['profiles'] = [p for p in data['profiles'] if p['driver_id'] != driver_id]
        
        # Add new profile
        data['profiles'].append(profile)
        
        with open(database_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[SUCCESS] Driver '{name}' enrolled successfully!")
        print(f"Database: {database_path}")
        print("="*70)
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enroll a driver")
    parser.add_argument("--id", type=str, required=True, help="Driver ID")
    parser.add_argument("--name", type=str, required=True, help="Driver name")
    parser.add_argument("--db", type=str, default="drivers.json", help="Database path")
    
    args = parser.parse_args()
    
    enroller = FaceEnroller()
    enroller.enroll_driver(args.id, args.name, args.db)
