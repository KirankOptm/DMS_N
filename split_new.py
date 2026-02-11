# dmsv8.py  — split-screen DMS + Seatbelt + Cigarette Smoking (CV-based)
import cv2
import mediapipe as mp
import time
import argparse
import numpy as np
from collections import deque
from datetime import datetime
from collections import defaultdict
import textwrap

# ---- Seatbelt module ----
from seatbelt_module import SeatBeltDetector

# =========================
# Args
# =========================
parser = argparse.ArgumentParser()
# (Your original DMS args)
parser.add_argument('--ear_threshold', type=float, default=0.180)
parser.add_argument('--eye_closed_frames_threshold', type=int, default=9)
parser.add_argument('--blink_rate_threshold', type=int, default=5)
parser.add_argument('--mar_threshold', type=float, default=0.6)
parser.add_argument('--yawn_threshold', type=int, default=3)
parser.add_argument('--frame_width', type=int, default=640)
parser.add_argument('--frame_height', type=int, default=480)
parser.add_argument('--gaze_deviation_threshold', type=float, default=0.05)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--head_turn_threshold', type=float, default=0.08)
parser.add_argument('--hand_near_face_px', type=int, default=200)
parser.add_argument('--calibration_time', type=int, default=15)
parser.add_argument('--no_face_display', action='store_true')
parser.add_argument('--no_mesh_display', action='store_true')
parser.add_argument('--display_width', type=int, default=1024, help="Output window width (panel)")
parser.add_argument('--display_height', type=int, default=600, help="Output window height (panel)")
parser.add_argument('--fullscreen', action='store_true', help="OpenCV fullscreen window")

# ---- Seatbelt flags ----
parser.add_argument("--seatbelt", action="store_true")
parser.add_argument("--seatbelt_model", type=str, default="best.onnx")
parser.add_argument("--seatbelt_conf", type=float, default=0.25)
parser.add_argument("--seatbelt_iou", type=float, default=0.45)
parser.add_argument("--seatbelt_emit_interval", type=float, default=1.0)
parser.add_argument("--seatbelt_names", type=str, default="seatbelt_worn,no_seatbelt")
parser.add_argument("--sb_roi", type=str, default="box", choices=["box", "pose"])

# ---- Smoking flags (new) ----
parser.add_argument("--smoking", action="store_true",
                    help="Enable cigarette smoking detection (ember near mouth + hand-near-face)")
parser.add_argument("--smoke_glow_min_area", type=int, default=10,
                    help="Min ember (red/orange) area in mouth ROI to consider (pixels)")
parser.add_argument("--smoke_persist_frames", type=int, default=6,
                    help="Frames of persistent ember needed (temporal smoothing)")
parser.add_argument("--smoke_cooldown", type=float, default=2.0,
                    help="Seconds between smoking alerts")
parser.add_argument("--smoke_debug", action="store_true",
                    help="Draw mouth ROI and ember mask for debugging")

args = parser.parse_args()

# =========================
# Setup
# =========================
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
NOSE_TIP = 1
LEFT_EAR_TIP = 234
RIGHT_EAR_TIP = 454

# =========================
# Camera open (your proven pipelines kept)
# =========================
pipelines = [
    (
        "v4l2src device=/dev/video3 io-mode=dmabuf ! "
        f"video/x-raw,format=YUY2,width={args.frame_width},height={args.frame_height},framerate=30/1 ! "
        "imxvideoconvert_g2d ! "
        "video/x-raw,format=BGR ! "
        "queue leaky=upstream max-size-buffers=1 ! "
        "appsink drop=true sync=false max-buffers=1"
    ),
    (
        "v4l2src device=/dev/video3 io-mode=dmabuf ! "
        f"video/x-raw,format=YUY2,width={args.frame_width},height={args.frame_height},framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "queue leaky=upstream max-size-buffers=1 ! "
        "appsink drop=true sync=false max-buffers=1"
    ),
    (
        "v4l2src device=/dev/video3 ! "
        f"video/x-raw,width={args.frame_width},height={args.frame_height},framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "queue leaky=upstream max-size-buffers=1 ! "
        "appsink drop=true sync=false max-buffers=1"
    ),
]
cap = None
for pipe in pipelines:
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[camera] Opened via GStreamer:", pipe)
        break
if cap is None or not cap.isOpened():
    print("[camera] Falling back to V4L2 path /dev/video4")
    cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.frame_width * args.scale_factor))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.frame_height * args.scale_factor))
    cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =========================
# Rolling display logs (right panel)
# =========================
display_logs = deque(maxlen=28)

def push_log(text):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"{ts}  {text}"
    display_logs.append(line)
    print(line, flush=True)

# =========================
# Seatbelt detector
# =========================
sb_detector = None
last_sb_emit_ts = 0.0
last_sb_state = "unknown"
if args.seatbelt:
    class_names = tuple([s.strip() for s in args.seatbelt_names.split(",") if s.strip()])
    if len(class_names) < 2:
        class_names = ("seatbelt_worn", "no_seatbelt")
    sb_detector = SeatBeltDetector(
        model_path=args.seatbelt_model,
        names=class_names,
        conf_thres=args.seatbelt_conf,
        iou_thres=args.seatbelt_iou,
        roi_mode=args.sb_roi
    )
    push_log(f"[Seatbelt] model={args.seatbelt_model} ROI={args.sb_roi} names={class_names}")

# =========================
# Smoking detector (CV-based)
# =========================
smoking_enabled = args.smoking
smoke_counter = 0
last_smoke_emit_ts = 0.0
SMOKE_ALERT_TEXT = "Possible CIGARETTE SMOKING detected"

def mouth_roi_from_landmarks(landmarks, w, h, expand=1.4):
    """Tight mouth bbox (from 13,14,78,308) expanded a bit."""
    idxs = MOUTH
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    x1, x2 = int(max(0, min(xs))), int(min(w - 1, max(xs)))
    y1, y2 = int(max(0, min(ys))), int(min(h - 1, max(ys)))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = int((x2 - x1) * expand), int((y2 - y1) * expand)
    x1e, y1e = max(0, cx - bw // 2), max(0, cy - bh // 2)
    x2e, y2e = min(w - 1, cx + bw // 2), min(h - 1, cy + bh // 2)
    return x1e, y1e, x2e, y2e

def detect_ember_in_roi(frame, roi_rect, min_area=10, debug=False):
    """Detect small bright orange/red regions (cigarette ember) inside ROI."""
    x1, y1, x2, y2 = roi_rect
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or (x2 - x1) < 5 or (y2 - y1) < 5:
        return False, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # two hue bands: red (wraps) and orange
    # red band1 (0-10)
    lower1 = np.array([0, 140, 180], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    # red band2 (170-179)
    lower2 = np.array([170, 130, 160], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    # orange (good for ember glow)
    lower3 = np.array([10, 120, 200], dtype=np.uint8)
    upper3 = np.array([25, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask3 = cv2.inRange(hsv, lower3, upper3)
    mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

    # noise cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # find blobs
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ember = False
    ember_box = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # reject overly big patches (just in case)
        if w * h > 0.15 * (roi.shape[0] * roi.shape[1]):
            continue
        ember = True
        ember_box = (x1 + x, y1 + y, w, h)
        break

    if debug:
        # draw ROI & mask preview in-frame corner
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 255, 255), 1)
        if ember_box:
            ex, ey, ew, eh = ember_box
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 165, 255), 2)
        # tiny mask preview (bottom-right of ROI)
        thumb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        th = max(20, (y2 - y1) // 3)
        tw = max(20, (x2 - x1) // 3)
        thumb = cv2.resize(thumb, (tw, th))
        oy = min(y2 - th, y1 + 4)
        ox = min(x2 - tw, x1 + 4)
        frame[oy:oy + th, ox:ox + tw] = thumb

    return ember, ember_box

# =========================
# Original counters/state
# =========================
eye_closure_counter = 0
blink_counter = 0
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

def add_alert(frame, message):
    ts = datetime.now().strftime("%H:%M:%S")
    active_alerts[f"{ts} {message}"] = time.time()
    #push_log(f"[ALERT] {message}")
    return message

def get_aspect_ratio(landmarks, eye_indices, w, h):
    def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])
    A = np.linalg.norm(pt(eye_indices[1]) - pt(eye_indices[5]))
    B = np.linalg.norm(pt(eye_indices[2]) - pt(eye_indices[4]))
    C = np.linalg.norm(pt(eye_indices[0]) - pt(eye_indices[3]))
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

def calculate_forward_gaze_line(iris_center, gaze_vec, iris_radius):
    if gaze_vec is None:
        return None
    vx, vy = gaze_vec
    length = float(iris_radius) * 3.0
    x0, y0 = int(iris_center[0]), int(iris_center[1])
    x1, y1 = int(x0 + vx * length), int(y0 + vy * length)
    return (x0, y0), (x1, y1)

def get_iris_center_and_radius(landmarks, iris_indices, w, h):
    center = get_iris_center(landmarks, iris_indices, w, h)
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in iris_indices], dtype=np.float32)
    if pts.size == 0:
        return (int(center[0]), int(center[1])), 0.0
    dists = np.linalg.norm(pts - center, axis=1)
    r = float(np.mean(dists)) if len(dists) else 0.0
    return (int(center[0]), int(center[1])), r

def estimate_gaze_direction(landmarks, w, h):
    try:
        left_c1, left_c2 = 33, 133
        right_c1, right_c2 = 362, 263
        def xy(i):
            return np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=np.float32)
        lc1, lc2 = xy(left_c1), xy(left_c2)
        left_center = (lc1 + lc2) / 2.0
        left_w = np.linalg.norm(lc2 - lc1) + 1e-6
        lv1, lv2 = xy(160), xy(144)
        left_h = np.linalg.norm(lv2 - lv1) + 1e-6
        rc1, rc2 = xy(right_c1), xy(right_c2)
        right_center = (rc1 + rc2) / 2.0
        right_w = np.linalg.norm(rc2 - rc1) + 1e-6
        rv1, rv2 = xy(385), xy(380)
        right_h = np.linalg.norm(rv2 - rv1) + 1e-6
        li = np.array(get_iris_center(landmarks, LEFT_IRIS, w, h), dtype=np.float32)
        ri = np.array(get_iris_center(landmarks, RIGHT_IRIS, w, h), dtype=np.float32)
        ldx, ldy = (li - left_center) / np.array([left_w, left_h])
        rdx, rdy = (ri - right_center) / np.array([right_w, right_h])
        dx = float((ldx + rdx) / 2.0)
        dy = float((ldy + rdy) / 2.0)
        dx = max(min(dx, 1.0), -1.0)
        dy = max(min(dy, 1.0), -1.0)
        return (dx, dy), 1.0
    except Exception:
        return None, 0.0

eye_closed = 0
head_turn = 0
hands_free = False
head_tilt = 0
head_droop = 0
yawn = False
msg = ""
last_msg = "Normal and Active Driving"
fid = 0

# =========================
# Prepare window + layout
# =========================
window_name = "Drowsiness Monitor (Split)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
if args.fullscreen:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.resizeWindow(window_name, args.display_width, args.display_height)

# Layout: left = video, right = logs
left_w = int(args.display_width * 0.64)
right_w = args.display_width - left_w
panel_h = args.display_height

push_log("App started")
push_log("Split-screen UI ready")

# =========================
# Main Loop
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)
    if args.no_face_display:
        frame = np.zeros_like(frame)
    current_time = time.time()

    # -------- Face logic (existing) --------
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        face_center = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))

        left_ear = get_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = get_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2

        msg = last_msg
        msg = "Normal and Active Driving"
        eye_closed = 0
        head_turn = 0
        hands_free = False
        head_tilt = 0
        head_droop = 0
        yawn = False

        visible_iris_points = [
            idx for idx in LEFT_IRIS + RIGHT_IRIS
            if 0 <= idx < len(landmarks) and hasattr(landmarks[idx], 'visibility') and landmarks[idx].visibility > 0.1
        ]
        iris_visible = len(visible_iris_points) >= 4

        left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
        right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
        iris_center_avg = (left_iris + right_iris) / 2

        iris_y_avg = iris_center_avg[1] / h if 'iris_center_avg' in locals() else 0.5
        iris_missing_or_low = (not iris_visible) or (iris_y_avg > 0.5)
        eye_closed_by_ear = avg_ear < args.ear_threshold
        if eye_closed_by_ear and iris_missing_or_low:
            eye_closure_counter += 1
            if eye_closure_counter > 30:
                msg = "Alert: Eyes Closed Too Long"; msg = add_alert(frame, msg); last_msg = msg; eye_closed = 2
            elif eye_closure_counter > args.eye_closed_frames_threshold:
                msg = "Warning: Eyes Closed"; msg = add_alert(frame, msg); last_msg = msg; eye_closed = 1
        else:
            if 2 <= eye_closure_counter < args.eye_closed_frames_threshold:
                blink_counter += 1
            eye_closure_counter = 0

        if current_time - blink_timer > 60:
            if blink_counter >= args.blink_rate_threshold:
                msg = "High Blinking Rate"; msg = add_alert(frame, msg); last_msg = msg
            blink_counter = 0
            blink_timer = current_time

        mar = get_mar(landmarks, MOUTH, w, h)
        mar_deque.append(mar)
        if mar > args.mar_threshold:
            yawn_counter += 1
        if yawn_counter > args.yawn_threshold:
            msg = "Warning: Yawning"; msg = add_alert(frame, msg); last_msg = msg; yawn = True; yawn_counter = 0

        left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
        right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
        iris_center_avg = (left_iris + right_iris) / 2
        gaze_x_norm = iris_center_avg[0] / w

        # (Optional) draw gaze lines
        try:
            left_center, left_r = get_iris_center_and_radius(landmarks, LEFT_IRIS, w, h)
            right_center, right_r = get_iris_center_and_radius(landmarks, RIGHT_IRIS, w, h)
            gaze_vec, _ = estimate_gaze_direction(landmarks, w, h)
            if gaze_vec is not None:
                ls, le = calculate_forward_gaze_line(left_center, gaze_vec, left_r)
                rs, re = calculate_forward_gaze_line(right_center, gaze_vec, right_r)
                if ls and le: cv2.line(frame, ls, le, (0, 165, 255), 2)
                if rs and re: cv2.line(frame, rs, re, (0, 165, 255), 2)
            cv2.circle(frame, left_center, 2, (0, 0, 255), -1)
            cv2.circle(frame, right_center, 2, (0, 0, 255), -1)
        except Exception:
            pass

        head_x = landmarks[NOSE_TIP].x
        head_y = landmarks[NOSE_TIP].y

        if calibration_mode:
            cv2.putText(frame, "Please align your face naturally, facing forward, before the countdown ends",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            gaze_offset = abs(gaze_x_norm - gaze_center)
            head_x_offset = abs(head_x - head_center_x)
            head_y_offset = abs(head_y - head_center_y)

            if gaze_offset > args.gaze_deviation_threshold:
                if gaze_offset < 0.1:
                    msg = "Mild Gaze Deviation"; msg = add_alert(frame, msg); last_msg = msg
                elif gaze_offset < 0.2:
                    msg = "Moderate Gaze Deviation"; msg = add_alert(frame, msg); last_msg = msg
                else:
                    msg = "Severe Gaze Deviation"; msg = add_alert(frame, msg); last_msg = msg

            if head_x_offset > args.head_turn_threshold:
                if head_x_offset < 0.1:
                    msg = "Mild Head Turn"; msg = add_alert(frame, msg); last_msg = msg; head_turn = 1
                elif head_x_offset < 0.2:
                    msg = "Moderate Head Turn"; msg = add_alert(frame, msg); last_msg = msg; head_turn = 2
                else:
                    msg = "Severe Head Turn"; msg = add_alert(frame, msg); last_msg = msg; head_turn = 3

            if abs(head_y_offset) > args.head_turn_threshold:
                if head_y < head_center_y:
                    if abs(head_y_offset) < 0.08:
                        msg = "Mild Looking Upward"; last_msg = msg; head_tilt = 1
                    elif abs(head_y_offset) < 0.13:
                        msg = "Moderate Looking Upward"; last_msg = msg; head_tilt = 2
                    else:
                        msg = "Looking Upward"; msg = add_alert(frame, msg); last_msg = msg; head_tilt = 3
                else:
                    if abs(head_y_offset) < 0.07:
                        msg = "Head drooping symptom"; msg = add_alert(frame, msg); last_msg = msg; head_droop = 1
                    elif abs(head_y_offset) < 0.12:
                        msg = "Head drooping started"; msg = add_alert(frame, msg); last_msg = msg; head_droop = 2
                    else:
                        msg = "Head drooped"; msg = add_alert(frame, msg); last_msg = msg; head_droop = 3

        if not args.no_mesh_display:
            mp_drawing.draw_landmarks(
                frame, result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

        # -------- Smoking detection (new) --------
        if smoking_enabled:
            mx1, my1, mx2, my2 = mouth_roi_from_landmarks(landmarks, w, h, expand=1.5)
            ember, ebox = detect_ember_in_roi(
                frame, (mx1, my1, mx2, my2),
                min_area=args.smoke_glow_min_area,
                debug=args.smoke_debug
            )

            # hand-near-face check (from your existing hands pipeline)
            hand_near = False
            if hand_result.multi_hand_landmarks:
                for hLM in hand_result.multi_hand_landmarks:
                    if hand_near_face((int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h)), hLM, frame.shape):
                        hand_near = True
                        break

            # temporal gating
            if ember or hand_near:
                smoke_counter += 1
            else:
                smoke_counter = max(0, smoke_counter - 1)

            # emit alert if persistent & cooldown passed
            if smoke_counter >= args.smoke_persist_frames:
                if (time.time() - last_smoke_emit_ts) >= args.smoke_cooldown:
                    add_alert(frame, SMOKE_ALERT_TEXT)
                    #push_log("SMOKING: ember/hand signal near mouth")
                    last_smoke_emit_ts = time.time()
                smoke_counter = args.smoke_persist_frames  # clamp (keeps 'armed' but not growing)

            # visualize mouth ROI for operator awareness
            if args.smoke_debug:
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 215, 255), 1)

    # -------- Hand logic (existing) --------
    if hand_result.multi_hand_landmarks:
        hand_coords = []
        for hand_landmarks in hand_result.multi_hand_landmarks:
            if result.multi_face_landmarks:
                if hand_near_ear(landmarks, hand_landmarks, w, h):
                    msg = "Likely mobile call"; msg = add_alert(frame, msg); last_msg = msg; hands_free = True
                elif hand_near_face((int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h)), hand_landmarks, frame.shape):
                    msg = "Hand near the face"; msg = add_alert(frame, msg); last_msg = msg; hands_free = True

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            hand_coords.append((np.mean(xs), np.mean(ys)))
            if not args.no_mesh_display:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if (not calibration_mode) and len(hand_coords) == 2:
            (x1, y1), (x2, y2) = hand_coords
            dist = np.hypot(x2 - x1, y2 - y1)
            both_hands_low = y1 > 0.6 and y2 > 0.6
            not_near_ear = not hand_near_ear(landmarks, hand_result.multi_hand_landmarks[0], w, h) and \
                           not hand_near_ear(landmarks, hand_result.multi_hand_landmarks[1], w, h)
            if dist < 0.35 and both_hands_low and not_near_ear:
                msg = "Possible texting observed"; msg = add_alert(frame, msg); last_msg = msg; hands_free = True

    # -------- Drowsiness/Distraction summary (existing) --------
    if (eye_closed == 2 and head_droop >= 1) or (eye_closed == 2 and yawn):
        msg = "Severe DROWSINESS Observed"; msg = add_alert(frame, msg); last_msg = msg
    elif (eye_closed == 1 and head_droop >= 1) or (eye_closed == 1 and yawn):
        msg = "Moderate DROWSINESS Observed"; msg = add_alert(frame, msg); last_msg = msg
    if (head_turn >= 1 and hands_free) or (head_tilt >= 1 and hands_free):
        msg = "Moderate DISTRACTION Observed"; msg = add_alert(frame, msg); last_msg = msg
    elif (head_turn >= 2 and hands_free) or (head_tilt >= 2 and hands_free):
        msg = "Severe DISTRACTION Observed"; msg = add_alert(frame, msg); last_msg = msg

    # -------- Seatbelt detection (existing) --------
    if sb_detector is not None:
        sb_status, frame, sb_meta = sb_detector.infer(frame, draw=True)
        now_emit = time.time()
        if (sb_status != last_sb_state) and ((now_emit - last_sb_emit_ts) >= args.seatbelt_emit_interval):
            if sb_status == "no_seatbelt":
                #push_log("SEATBELT ALERT: Not detected")
                add_alert(frame, "Seatbelt NOT detected")
            elif sb_status == "seatbelt_worn":
                push_log("SEATBELT OK")
                add_alert(frame, "Seatbelt OK")
            last_sb_state = sb_status
            last_sb_emit_ts = now_emit

    # Expire alerts
    expired = [k for k, t in active_alerts.items() if current_time - t > ALERT_DURATION]
    for k in expired:
        del active_alerts[k]

    # ===== Split-screen composition =====
    canvas = np.zeros((panel_h, args.display_width, 3), dtype=np.uint8)

    # Left: video (fit)
    scale = min(left_w / w, panel_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    x0 = (left_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    # Right: logs
    log_x0 = left_w
    cv2.rectangle(canvas, (log_x0, 0), (log_x0 + right_w - 1, panel_h - 1), (30, 30, 30), -1)
    cv2.putText(canvas, "EVENT LOGS", (log_x0 + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Active alerts
    y = 60
    for msg_txt in list(active_alerts.keys())[-6:]:
        color = (0,0,255) if "Severe" in msg_txt else ((0,255,255) if "Moderate" in msg_txt or "Alert" in msg_txt else (255,255,255))
        cv2.putText(canvas, msg_txt, (log_x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 22
    cv2.line(canvas, (log_x0 + 8, y+6), (log_x0 + right_w - 8, y+6), (80, 80, 80), 1)
    y += 28

    # Rolling logs
    max_cols = max(20, (right_w - 24) // 8)
    for raw in list(display_logs)[-int((panel_h - y) / 20):]:
        for seg in textwrap.wrap(raw, width=max_cols):
            if y >= panel_h - 10: break
            cv2.putText(canvas, seg, (log_x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y += 20

    # Calibration overlay
    if calibration_mode:
        countdown = int(calibration_duration - (time.time() - calibration_start_time))
    if calibration_mode and countdown > 0:
        cv2.putText(canvas, f"Calibrating in {countdown}s...", (20, panel_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        key = cv2.waitKey(1) & 0xFF
    if calibration_mode and countdown == 0 and result.multi_face_landmarks:
        gaze_center = gaze_x_norm
        head_center_x = landmarks[NOSE_TIP].x
        head_center_y = landmarks[NOSE_TIP].y
        calibration_mode = False
        push_log("Calibration Complete")
        countdown = -1

    cv2.imshow(window_name, canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

