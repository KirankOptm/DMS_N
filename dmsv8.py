import cv2
import mediapipe as mp
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


# ============================================================
# MJPEG Streaming Server for VLC viewing
# ============================================================re

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
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--ear_threshold', type=float, default=0.180)
parser.add_argument('--eye_closed_frames_threshold', type=int, default=9)
parser.add_argument('--blink_rate_threshold', type=int, default=5)
parser.add_argument('--mar_threshold', type=float, default=0.6)
parser.add_argument('--yawn_threshold', type=int, default=3)
parser.add_argument('--frame_width', type=int, default=1280)
parser.add_argument('--frame_height', type=int, default=720)
parser.add_argument('--gaze_deviation_threshold', type=float, default=0.05, help="Threshold for gaze deviation from center (0 to 1)")
parser.add_argument('--scale_factor', type=float, default=1.0, help="Scaling factor for resolution (e.g., 0.5x, 1x, 2x)")
parser.add_argument('--head_turn_threshold', type=float, default=0.08, help="Threshold for head turning detection")
parser.add_argument('--hand_near_face_px', type=int, default=200, help="Distance threshold for hand near face detection")
parser.add_argument('--calibration_time', type=int, default=15, help="Calibration duration in seconds")
parser.add_argument('--no_face_display', action='store_true', help="Disable face and mesh drawing on frame")
parser.add_argument('--no_mesh_display', action='store_true', help="Disable face and mesh drawing on frame")
parser.add_argument('--mjpeg_port', type=int, default=8080, help="MJPEG streaming port for VLC viewing")
parser.add_argument('--enable_streaming', action='store_true', default=True, help="Enable MJPEG streaming (default: enabled)")
parser.add_argument('--headless', action='store_true', default=True, help="Run without display window (default: enabled for embedded boards)")
parser.add_argument('--camera_device', type=str, default='auto', help="Camera device (e.g., /dev/video0, 0, or 'auto' for auto-detect)")
parser.add_argument('--crop_center', type=float, default=0.0, help="Center crop percentage (0.0-0.5). E.g., 0.3 crops 30%% from each edge")

args = parser.parse_args()

# Setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
NOSE_TIP = 1  # Nose landmark index
LEFT_EAR_TIP = 234
RIGHT_EAR_TIP = 454

# Auto-detect available camera
def find_available_camera():
    """Scan for available cameras and return first working one"""
    print("[Camera] Auto-detecting available cameras...")
    
    # First try /dev/video* devices (Linux)
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
    
    # Fallback: try numeric indices 0-5
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
    # Auto-detect or use specified device
    if args.camera_device == 'auto':
        device_num = find_available_camera()
        if device_num is None:
            print("[ERROR] No camera available")
            print("[INFO] Try: ls -l /dev/video* or specify manually with --camera_device")
            exit(1)
    else:
        # Extract device number from /dev/videoX or use directly
        if '/dev/video' in args.camera_device:
            device_num = int(args.camera_device.split('video')[-1])
        else:
            device_num = int(args.camera_device)
    
    print(f"[Camera] Opening camera {device_num}...")
    cap = cv2.VideoCapture(device_num)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Don't set resolution - use camera's native resolution (2MP)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {device_num}")
        print("[INFO] Try: ls -l /dev/video* to find your camera"        cd /home/dms_imx93_c
        ./build.sh)
        print("[INFO] Available cameras: 0, 1, 2, 3...")
        exit(1)
    
    print("[Camera] Opened successfully")
except Exception as e:
    print(f"[ERROR] Camera initialization failed: {e}")
    exit(1)

# Start MJPEG streaming server if enabled
mjpeg_server = None
if args.enable_streaming:
    mjpeg_server = MJPEGServer(host='0.0.0.0', port=args.mjpeg_port)
    if mjpeg_server.start():
        print(f"[DMS] Stream ready at port {args.mjpeg_port}")
    else:
        mjpeg_server = None

#cap = cv2.VideoCapture(4)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.frame_width * args.scale_factor))
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.frame_height * args.scale_factor))

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

def get_aspect_ratio(landmarks, eye_indices, w, h):
    def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])

    A = np.linalg.norm(pt(eye_indices[1]) - pt(eye_indices[5]))  # vertical
    B = np.linalg.norm(pt(eye_indices[2]) - pt(eye_indices[4]))  # vertical
    C = np.linalg.norm(pt(eye_indices[0]) - pt(eye_indices[3]))  # horizontal

    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear

def hand_near_ear(landmarks, hand_landmarks, w, h):
    ear_l = np.array([landmarks[LEFT_EAR_TIP].x * w, landmarks[LEFT_EAR_TIP].y * h])
    ear_r = np.array([landmarks[RIGHT_EAR_TIP].x * w, landmarks[RIGHT_EAR_TIP].y * h])
    for lm in hand_landmarks.landmark:
        hx, hy = lm.x * w, lm.y * h
        dx_l, dy_l = abs(hx - ear_l[0]), abs(hy - ear_l[1])
        dx_r, dy_r = abs(hx - ear_r[0]), abs(hy - ear_r[1])

        if (dx_l < 40 and dy_l < 90) or (dx_r < 40 and dy_r < 90):
            return True
    return False


def hand_near_face(face_center, hand_landmarks, shape):
    fcx, fcy = face_center
    ih, iw = shape[:2]
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * iw), int(lm.y * ih)
        if np.hypot(fcx - x, fcy - y) < args.hand_near_face_px:
            return True
    return False


def get_mar(landmarks, mouth_idx, w, h):
    top = np.array([landmarks[mouth_idx[0]].x * w, landmarks[mouth_idx[0]].y * h])
    bottom = np.array([landmarks[mouth_idx[1]].x * w, landmarks[mouth_idx[1]].y * h])
    left = np.array([landmarks[mouth_idx[2]].x * w, landmarks[mouth_idx[2]].y * h])
    right = np.array([landmarks[mouth_idx[3]].x * w, landmarks[mouth_idx[3]].y * h])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal if horizontal > 0 else 0

def get_iris_center(landmarks, indices, w, h):
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    return np.mean(points, axis=0)
    

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
    
    # Crop center portion for wide-angle cameras (reduces distortion)
    if args.crop_center > 0 and args.crop_center <= 0.5:
        h_orig, w_orig = frame.shape[:2]
        crop_h = int(h_orig * args.crop_center)
        crop_w = int(w_orig * args.crop_center)
        frame = frame[crop_h:h_orig-crop_h, crop_w:w_orig-crop_w]
        if fid == 1:
            print(f"[Camera] Cropped from {w_orig}x{h_orig} to {frame.shape[1]}x{frame.shape[0]}")
    
    # Log frame info on first frame
    if fid == 1:
        print(f"[Camera] Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"[Camera] Frame format: {frame.dtype}")
        print("[DMS] Processing started...")

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create display frame for UI
    display_frame = frame.copy()
    result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)
    
    # Log detection status every 30 frames
    if fid % 30 == 0:
        face_status = "Face detected" if result.multi_face_landmarks else "No face"
        hand_status = "Hand detected" if hand_result.multi_hand_landmarks else "No hand"
        print(f"[Frame {fid}] {face_status}, {hand_status}")
    
    if args.no_face_display:
        frame = np.zeros_like(frame)
    current_time = time.time()

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        face_center = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))
        #EYE CLOSURE
        left_ear = get_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = get_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2
        print ("AVG EAR = ",avg_ear)
        msg = last_msg
        msg = "Normal and Active Driving"
        eye_closed = 0
        head_turn = 0
        hands_free = False
        head_tilt = 0
        head_droop = 0
        yawn = False
        visible_ears = []
        if left_ear > 0: visible_ears.append(left_ear)
        if right_ear > 0: visible_ears.append(right_ear)

        #avg_ear = np.mean(visible_ears) if visible_ears else 1.0

        visible_iris_points = [
            idx for idx in LEFT_IRIS + RIGHT_IRIS
            if 0 <= idx < len(landmarks) and hasattr(landmarks[idx], 'visibility') and landmarks[idx].visibility > 0.1
        ]
        iris_visible = len(visible_iris_points) >= 4

        left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
        right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
        iris_center_avg = (left_iris + right_iris) / 2

        # Enhanced logic: eye is closed only if EAR low AND iris not visible (or below vertical threshold)
        iris_y_avg = iris_center_avg[1] / h if 'iris_center_avg' in locals() else 0.5
 	
        iris_missing_or_low = (not iris_visible) or (iris_y_avg > 0.5)  # Adjust 0.65 based on testing
        #print ("Iris missing or low",iris_missing_or_low)
        #print ("Iris visible and y_avg",iris_visible, iris_y_avg)
        eye_closed_by_ear = avg_ear < args.ear_threshold  # Lowered threshold
        #print("eye_closed_by_ear",eye_closed_by_ear)
        if eye_closed_by_ear and iris_missing_or_low:
            eye_closure_counter += 1
            #print("eye closure counter, eye threshold",eye_closure_counter,args.eye_closed_frames_threshold)

            if eye_closure_counter > 30:
                msg = "Alert: Eyes Closed Too Long"
                msg = add_alert(frame,msg)
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
                msg = add_alert(frame,msg)
                last_msg = msg
            blink_counter = 0
            blink_timer = current_time

        mar = get_mar(landmarks, MOUTH, w, h)
        mar_deque.append(mar)
        if mar > args.mar_threshold:
            yawn_counter += 1

        if yawn_counter > args.yawn_threshold:
            msg = "Warning: Yawning"
            msg = add_alert(frame,msg)
            last_msg = msg

            #add_alert("Yawning")
            yawn = True
            yawn_counter = 0

        # Gaze estimation
        left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
        right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
        iris_center_avg = (left_iris + right_iris) / 2
        gaze_x_norm = iris_center_avg[0] / w

        # Head pose estimation (using nose tip)
        head_x = landmarks[NOSE_TIP].x
        head_y = landmarks[NOSE_TIP].y

        if calibration_mode:
            cv2.putText(frame, "Please align your face naturally, facing forward, before the countdown ends", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            gaze_offset = abs(gaze_x_norm - gaze_center)
            head_x_offset = abs(head_x - head_center_x)
            head_y_offset = abs(head_y - head_center_y)

            if gaze_offset > args.gaze_deviation_threshold:
                if gaze_offset < 0.1:
                    msg = "Mild Gaze Deviation"
                    msg = add_alert(frame,msg)
                    last_msg = msg
                elif gaze_offset < 0.2:
                    msg = "Moderate Gaze Deviation"
                    msg = add_alert(frame,msg)
                    last_msg = msg
                else:
                    msg = "Severe Gaze Deviation"
                    msg = add_alert(frame,msg)
                    last_msg = msg

            if head_x_offset > args.head_turn_threshold:
                if head_x_offset < 0.1:
                    msg = "Mild Head Turn"
                    msg = add_alert(frame,msg)
                    last_msg = msg
                    head_turn = 1
                elif head_x_offset < 0.2:
                    msg = "Moderate Head Turn"
                    msg = add_alert(frame,msg)
                    last_msg = msg
                    head_turn = 2
                else:
                    msg = "Severe Head Turn"
                    msg = add_alert(frame,msg)
                    last_msg = msg
                    head_turn = 3
            """
            if head_y_offset > args.head_turn_threshold:
                if head_y_offset < 0.1:
                    add_alert("Mild Head Tilt")
                elif head_y_offset < 0.2:
                    add_alert("Moderate Head Tilt")
                else:
                    add_alert("Severe Head Tilt")
             """

            # Detect Upward vs Downward head tilt
            if abs(head_y_offset) > args.head_turn_threshold:
                if head_y < head_center_y:
                # Head tilted upward
                    if abs(head_y_offset) < 0.08:
                        msg = "Mild Looking Upward"
                        #msg = add_alert(frame,msg)
                        last_msg = msg
                        head_tilt = 1
                    elif abs(head_y_offset) < 0.13:
                        msg = "Moderate Looking Upward"
                        #msg = add_alert(frame,msg)
                        last_msg = msg
                        head_tilt = 2
                    else:
                        msg = "Looking Upward"
                        msg = add_alert(frame,msg)
                        last_msg = msg
                        head_tilt = 3
                else:
                # Head tilted downward
                    if abs(head_y_offset) < 0.07:
                        msg = "Head drooping symptom"
                        msg = add_alert(frame,msg)
                        last_msg = msg
                        head_droop = 1
                    elif abs(head_y_offset) < 0.12:
                        msg = "Head drooping started"
                        msg = add_alert(frame,msg)
                        last_msg = msg
                        head_droop = 2
                    else:
                        msg = "Head drooped"
                        msg = add_alert(frame,msg)
                        last_msg = msg
                        head_droop = 3
 
        if not args.no_mesh_display: mp_drawing.draw_landmarks(frame, result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

    # Hand detection and proximity alert
    if hand_result.multi_hand_landmarks:
        hand_coords = []
        for hand_landmarks in hand_result.multi_hand_landmarks:
            if result.multi_face_landmarks:
                if hand_near_ear(landmarks, hand_landmarks, w, h):
                    msg = "Likely mobile call"
                    msg = add_alert(frame,msg)
                    last_msg = msg
                    hands_free = True
                elif hand_near_face(face_center, hand_landmarks, frame.shape):
                    msg = "Hand near the face"
                    msg = add_alert(frame,msg)
                    last_msg = msg
                    hands_free = True

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            hand_coords.append((np.mean(xs), np.mean(ys)))
            if not args.no_mesh_display: mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        if calibration_mode is False and len(hand_coords) == 2:
            (x1, y1), (x2, y2) = hand_coords
            dist = np.hypot(x2 - x1, y2 - y1)

            both_hands_low = y1 > 0.6 and y2 > 0.6
            not_near_ear = not hand_near_ear(landmarks, hand_result.multi_hand_landmarks[0], w, h) and \
                   not hand_near_ear(landmarks, hand_result.multi_hand_landmarks[1], w, h)

            if dist < 0.35 and both_hands_low and not_near_ear:
                msg = "Possible texting observed"
                msg = add_alert(frame,msg)
                last_msg = msg
                hands_free = True
    

    if eye_closed == 2 and head_droop >= 1 or eye_closed == 2 and yawn:
        msg = "Severe DROWSINESS Observed"
        msg = add_alert(frame,msg)
        last_msg = msg
    elif eye_closed == 1 and head_droop >= 1 or eye_closed == 1 and yawn:
        msg = "Moderate DROWSINESS Observed"
        msg = add_alert(frame,msg)
        last_msg = msg
    if head_turn >= 1 and hands_free or head_tilt >= 1  and hands_free:
        msg = "Moderate DISTRACTION Observed"
        msg = add_alert(frame,msg)
        last_msg = msg    
    elif head_turn >= 2 and hands_free or head_tilt >= 2 and hands_free:
        msg = "Severe DISTRACTION Observed"
        msg = add_alert(frame,msg)
        last_msg = msg


     # Expire alerts
    expired = [k for k, t in active_alerts.items() if current_time - t > ALERT_DURATION]
    for k in expired:
        del active_alerts[k]

    for i, msg in enumerate(active_alerts):
        if "Mild" in msg or  "Warning" in msg:
            color = (255, 255, 255)  # White
        elif "Moderate" in msg or "Alert" in msg:
            color = (0, 255, 255)    # Yellow
        elif "Severe" in msg:
            color = (0, 0, 255)      # Bright Red
        else:
            color = (0, 0, 255)      # Default Red
        cv2.putText(frame, msg, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Update MJPEG stream
    if mjpeg_server is not None:
        mjpeg_server.update_frame(frame)

    if not args.headless:
        cv2.imshow("Drowsiness Monitor", frame)
    #last_msg = msg
    if calibration_mode:
        countdown = int(calibration_duration - (time.time() - calibration_start_time))
    if countdown > 0:
        if countdown % 5 == 0 and fid % 30 == 0:  # Log every 5 seconds
            if result.multi_face_landmarks:
                print(f"[Calibration] {countdown}s remaining - Face detected OK")
            else:
                print(f"[Calibration] {countdown}s remaining - WARNING: No face detected!")
        cv2.putText(frame, f"Calibrating in {countdown}s...", (20, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not args.headless:
            cv2.imshow("Drowsiness Monitor", frame)
        key = cv2.waitKey(1) & 0xFF if not args.headless else 0xFF
    if countdown == 0 and result.multi_face_landmarks:
        # Trigger calibration once countdown finishes
        gaze_center = gaze_x_norm
        head_center_x = landmarks[NOSE_TIP].x
        head_center_y = landmarks[NOSE_TIP].y
        calibration_mode = False
        print(f"[Calibration] COMPLETE - Gaze center: {gaze_center:.3f}, Head center: ({head_center_x:.3f}, {head_center_y:.3f})")
        add_alert(frame, "Calibration Complete")
        countdown = -1
    elif countdown == 0 and not result.multi_face_landmarks:
        calibration_mode = False
        print(f"[Calibration] WARNING - Completed without face detection! Using defaults.")
        countdown = -1
    
    if not args.headless:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    else:
        time.sleep(0.001)  # Small delay in headless mode
    
    """
    elif key == ord('c') and result.multi_face_landmarks:
        gaze_center = gaze_x_norm
        head_center_x = landmarks[NOSE_TIP].x
        head_center_y = landmarks[NOSE_TIP].y
        calibration_mode = False
        print(f"Calibrated gaze center: {gaze_center:.3f}, head center: ({head_center_x:.3f}, {head_center_y:.3f})")
    """

print("\n[DMS] Shutting down...")
cap.release()
if not args.headless:
    cv2.destroyAllWindows()

# Stop MJPEG server
if mjpeg_server is not None:
    mjpeg_server.stop()

