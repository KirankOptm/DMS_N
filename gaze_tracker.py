#!/usr/bin/env python3
"""
Gaze Tracker - Detects eye gaze direction (left, right, up, down, center)
Uses MediaPipe iris landmarks to track eyeball movement independent of head pose
"""

import numpy as np
import cv2


class GazeTracker:
    """
    Tracks gaze direction using iris position relative to eye corners
    Works without head movement - tracks only eyeball position
    """
    
    # MediaPipe Face Mesh landmark indices
    LEFT_EYE_CORNERS = [33, 133]    # Inner, outer corner
    RIGHT_EYE_CORNERS = [362, 263]  # Inner, outer corner
    
    LEFT_IRIS = [468, 469, 470, 471, 472]   # 5 iris landmarks
    RIGHT_IRIS = [473, 474, 475, 476, 477]  # 5 iris landmarks
    
    def __init__(self, 
                 horizontal_threshold=0.15,
                 vertical_threshold=0.15,
                 smoothing_frames=3,
                 flip_horizontal=True,
                 flip_vertical=False,
                 horizontal_center=0.50,
                 vertical_center=0.50):
        """
        Initialize gaze tracker
        
        Args:
            horizontal_threshold: Ratio threshold for left/right detection (0.0-0.5)
            vertical_threshold: Ratio threshold for up/down detection (0.0-0.5)
            smoothing_frames: Number of frames to smooth gaze direction
            flip_horizontal: Flip left/right (True for front-facing camera)
            flip_vertical: Flip up/down (False for most cameras)
            horizontal_center: Center point for horizontal (default 0.50, adjust for side cameras)
            vertical_center: Center point for vertical (default 0.50, adjust for elevated cameras)
        """
        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold
        self.smoothing_frames = smoothing_frames
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.horizontal_center = horizontal_center
        self.vertical_center = vertical_center
        
        # Smoothing buffer
        self.gaze_history = []
        
        # Current gaze
        self.gaze_direction = "CENTER"
        self.horizontal_ratio = 0.5
        self.vertical_ratio = 0.5
    
    def get_iris_position(self, landmarks, iris_indices, eye_corners, frame_width, frame_height):
        """
        Calculate iris center position normalized relative to eye corners
        
        Returns:
            horizontal_ratio: 0.0 (left) to 1.0 (right)
            vertical_ratio: 0.0 (up) to 1.0 (down)
        """
        # Get iris center (average of 5 iris landmarks)
        iris_points = []
        for idx in iris_indices:
            lm = landmarks[idx]
            iris_points.append([lm.x * frame_width, lm.y * frame_height])
        iris_center = np.mean(iris_points, axis=0)
        
        # Get eye corners
        inner_corner = landmarks[eye_corners[0]]
        outer_corner = landmarks[eye_corners[1]]
        
        inner_point = np.array([inner_corner.x * frame_width, inner_corner.y * frame_height])
        outer_point = np.array([outer_corner.x * frame_width, outer_corner.y * frame_height])
        
        # Calculate eye width and height
        eye_width = np.linalg.norm(outer_point - inner_point)
        
        # Horizontal ratio: iris position between inner (0.0) and outer (1.0)
        iris_to_inner = np.linalg.norm(iris_center - inner_point)
        horizontal_ratio = iris_to_inner / (eye_width + 1e-6)
        
        # Vertical ratio: calculate vertical eye span
        # Use upper and lower eyelid landmarks for better accuracy
        if eye_corners == self.LEFT_EYE_CORNERS:
            # Left eye vertical landmarks
            upper_lid_idx = 159
            lower_lid_idx = 145
        else:
            # Right eye vertical landmarks
            upper_lid_idx = 386
            lower_lid_idx = 374
        
        upper_lid = landmarks[upper_lid_idx]
        lower_lid = landmarks[lower_lid_idx]
        
        upper_point = np.array([upper_lid.x * frame_width, upper_lid.y * frame_height])
        lower_point = np.array([lower_lid.x * frame_width, lower_lid.y * frame_height])
        
        eye_height = np.linalg.norm(lower_point - upper_point)
        iris_to_upper = np.linalg.norm(iris_center - upper_point)
        vertical_ratio = iris_to_upper / (eye_height + 1e-6)
        
        return horizontal_ratio, vertical_ratio
    
    def classify_gaze(self, horizontal_ratio, vertical_ratio):
        """
        Classify gaze direction based on iris position ratios
        
        Returns:
            gaze_direction: "LEFT", "RIGHT", "UP", "DOWN", "CENTER"
        """
        # Apply flips for camera orientation
        if self.flip_horizontal:
            horizontal_ratio = 1.0 - horizontal_ratio  # Invert horizontal
        if self.flip_vertical:
            vertical_ratio = 1.0 - vertical_ratio  # Invert vertical
        
        # Horizontal classification (use custom center point)
        if horizontal_ratio < (self.horizontal_center - self.horizontal_threshold):
            horizontal = "LEFT"
        elif horizontal_ratio > (self.horizontal_center + self.horizontal_threshold):
            horizontal = "RIGHT"
        else:
            horizontal = "CENTER_H"
        
        # Vertical classification (use custom center point)
        if vertical_ratio < (self.vertical_center - self.vertical_threshold):
            vertical = "UP"
        elif vertical_ratio > (self.vertical_center + self.vertical_threshold):
            vertical = "DOWN"
        else:
            vertical = "CENTER_V"
        
        # Combine horizontal and vertical (PRIORITIZE VERTICAL for UP/DOWN detection)
        # Check vertical first
        if vertical == "UP" and horizontal == "CENTER_H":
            return "UP"
        elif vertical == "DOWN" and horizontal == "CENTER_H":
            return "DOWN"
        # Then check horizontal
        elif horizontal == "LEFT" and vertical == "CENTER_V":
            return "LEFT"
        elif horizontal == "RIGHT" and vertical == "CENTER_V":
            return "RIGHT"
        # Both centered
        elif horizontal == "CENTER_H" and vertical == "CENTER_V":
            return "CENTER"
        else:
            # Diagonal or mixed - PRIORITIZE VERTICAL (more important for drowsiness)
            if vertical in ["UP", "DOWN"]:
                return vertical  # Always prefer UP/DOWN over LEFT/RIGHT
            else:
                return horizontal  # Only return horizontal if vertical is centered
    
    def process_frame(self, face_landmarks, frame_width, frame_height):
        """
        Process MediaPipe face landmarks to determine gaze direction
        
        Args:
            face_landmarks: MediaPipe face_landmarks object
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        
        Returns:
            gaze_direction: Current gaze direction string
            horizontal_ratio: RAW horizontal iris position BEFORE flip (0.0-1.0)
            vertical_ratio: RAW vertical iris position BEFORE flip (0.0-1.0)
        """
        if not face_landmarks:
            return self.gaze_direction, self.horizontal_ratio, self.vertical_ratio
        
        landmarks = face_landmarks.landmark
        
        # Calculate gaze for both eyes
        left_h, left_v = self.get_iris_position(
            landmarks, self.LEFT_IRIS, self.LEFT_EYE_CORNERS, 
            frame_width, frame_height
        )
        
        right_h, right_v = self.get_iris_position(
            landmarks, self.RIGHT_IRIS, self.RIGHT_EYE_CORNERS,
            frame_width, frame_height
        )
        
        # Average both eyes (RAW values)
        raw_horizontal = (left_h + right_h) / 2.0
        raw_vertical = (left_v + right_v) / 2.0
        
        # Store RAW values for return
        self.horizontal_ratio = raw_horizontal
        self.vertical_ratio = raw_vertical
        
        # Classify gaze (flipping happens inside classify_gaze)
        current_gaze = self.classify_gaze(raw_horizontal, raw_vertical)
        
        # Smooth with history
        self.gaze_history.append(current_gaze)
        if len(self.gaze_history) > self.smoothing_frames:
            self.gaze_history.pop(0)
        
        # Most common gaze in recent history
        if len(self.gaze_history) >= self.smoothing_frames:
            from collections import Counter
            self.gaze_direction = Counter(self.gaze_history).most_common(1)[0][0]
        else:
            self.gaze_direction = current_gaze
        
        return self.gaze_direction, self.horizontal_ratio, self.vertical_ratio
    
    def draw_gaze_overlay(self, frame, face_landmarks, frame_width, frame_height):
        """
        Draw gaze visualization on frame
        
        Args:
            frame: OpenCV frame (BGR)
            face_landmarks: MediaPipe face_landmarks
            frame_width: Frame width
            frame_height: Frame height
        """
        if not face_landmarks:
            return frame
        
        landmarks = face_landmarks.landmark
        
        # Draw iris circles
        for iris_indices in [self.LEFT_IRIS, self.RIGHT_IRIS]:
            iris_points = []
            for idx in iris_indices:
                lm = landmarks[idx]
                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)
                iris_points.append([x, y])
            
            iris_center = np.mean(iris_points, axis=0).astype(int)
            cv2.circle(frame, tuple(iris_center), 3, (0, 255, 0), -1)
        
        # Draw gaze direction text
        gaze_text = f"Gaze: {self.gaze_direction}"
        ratio_text = f"H:{self.horizontal_ratio:.2f} V:{self.vertical_ratio:.2f}"
        
        cv2.putText(frame, gaze_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, ratio_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def is_looking_away(self):
        """Check if driver is looking away from road (not CENTER)"""
        return self.gaze_direction != "CENTER"
    
    def get_gaze_info(self):
        """
        Get current gaze information as dictionary
        
        Returns:
            dict with gaze_direction, horizontal_ratio, vertical_ratio
        """
        return {
            'gaze_direction': self.gaze_direction,
            'horizontal_ratio': self.horizontal_ratio,
            'vertical_ratio': self.vertical_ratio,
            'looking_away': self.is_looking_away()
        }


# Test standalone
if __name__ == "__main__":
    import mediapipe as mp
    
    print("=" * 70)
    print("GAZE TRACKER TEST")
    print("=" * 70)
    print("Look LEFT, RIGHT, UP, DOWN with your eyes (without moving head)")
    print("Press 'q' to quit")
    print("=" * 70)
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # Required for iris tracking
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize gaze tracker
    gaze_tracker = GazeTracker(
        horizontal_threshold=0.15,
        vertical_threshold=0.15,
        smoothing_frames=3
    )
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        result = face_mesh.process(rgb)
        
        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            
            # Process gaze
            gaze_dir, h_ratio, v_ratio = gaze_tracker.process_frame(
                face_landmarks, w, h
            )
            
            # Draw overlay
            frame = gaze_tracker.draw_gaze_overlay(frame, face_landmarks, w, h)
            
            # Print status
            print(f"\rGaze: {gaze_dir:8s} | H: {h_ratio:.2f} | V: {v_ratio:.2f}", end="")
        
        cv2.imshow("Gaze Tracker Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
