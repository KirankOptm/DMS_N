"""
Integration Guide: Adding Latency Logging to dms_integrated_mdp_yolo.py

This file shows the key integration points. DO NOT run this file directly.
Copy these snippets into your main dms_integrated_mdp_yolo.py
"""

# ============================================================================
# STEP 1: Add import at top of file (after other imports)
# ============================================================================
"""
import dms_latency_logger as latency_logger
"""

# ============================================================================
# STEP 2: Add argparse argument (in argument parser section around line 850)
# ============================================================================
"""
ap.add_argument('--enable_latency_logs', action='store_true', 
                help='Enable latency logging for performance analysis')
ap.add_argument('--latency_log_path', type=str, default='/tmp/dms_latency.jsonl',
                help='Path to latency log file (JSONL format)')
"""

# ============================================================================
# STEP 3: Initialize logger (after args parsing, before main loop around line 1100)
# ============================================================================
"""
# Initialize latency logger
logger = None
if args.enable_latency_logs:
    logger = latency_logger.init_logger(
        log_path=args.latency_log_path,
        enabled=True
    )
    print(f"[DMS] Latency logging enabled → {args.latency_log_path}")
else:
    logger = latency_logger.init_logger(enabled=False)
"""

# ============================================================================
# STEP 4: Add timing points in main loop (while cap.isOpened():)
# ============================================================================

# At start of frame processing (after cap.read())
"""
    # START FRAME TIMING
    if logger:
        logger.start_frame(fid)
        logger.mark_stage('cap_read', start=True)
    
    ret, frame = cap.read()
    
    if logger:
        logger.mark_stage('cap_read', start=False)
    
    if not ret:
        if logger:
            logger.mark_dropped_frame()
        break
"""

# Before preprocessing (RGB conversion)
"""
    # PREPROCESS TIMING
    if logger:
        logger.mark_stage('preproc', start=True)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if logger:
        logger.mark_stage('preproc', start=False)
"""

# Before MediaPipe processing
"""
    # MEDIAPIPE TIMING
    if logger:
        logger.mark_stage('mediapipe', start=True)
    
    result = face_mesh.process(rgb) if face_mesh is not None else None
    hand_result = hands.process(rgb) if hands is not None else None
    pose_result = pose.process(rgb) if pose is not None else None
    
    if logger:
        logger.mark_stage('mediapipe', start=False)
"""

# Before YOLO inference
"""
    # YOLO TIMING
    in_burst = time.time() < burst_until_ts
    if yolo_worker and yolo_active and ((fid % max(1, (1 if in_burst else dynamic_skip))) == 0):
        if logger:
            logger.mark_stage('yolo', start=True)
        
        yolo_worker.submit(frame)
        
        if logger:
            logger.mark_stage('yolo', start=False)
"""

# ============================================================================
# STEP 5: Log MediaPipe alerts (when alerts are generated)
# ============================================================================

# Eye closure detection (around line 1750)
"""
    if eye_closed_by_ear and iris_missing_or_low:
        eye_closure_counter += 1
        if eye_closure_counter > 30:
            t_detection = logger.get_timestamp_ns() if logger else 0
            add_alert("Alert: Eyes Closed Too Long")
            eye_closed = 2
            
            # LOG MEDIAPIPE ALERT
            if logger:
                logger.log_mediapipe_alert(
                    alert_type='eye_closed_long',
                    detection_time_ns=t_detection,
                    duration_frames=eye_closure_counter,
                    avg_ear=avg_ear
                )
"""

# Yawning detection (around line 1775)
"""
    if yawn_counter > args.yawn_threshold:
        t_detection = logger.get_timestamp_ns() if logger else 0
        add_alert("Warning: Yawning")
        yawn = True
        yawn_counter = 0
        kss_calculator.add_yawn(current_time)
        
        t_buzzer = logger.get_timestamp_ns() if logger else 0
        buzzer_beep(times=1, on_s=0.1, off_s=0.0)
        
        # LOG MEDIAPIPE ALERT
        if logger:
            logger.log_mediapipe_alert(
                alert_type='yawn',
                detection_time_ns=t_detection,
                action_time_ns=t_buzzer,
                mar=mar
            )
"""

# Head droop detection (around line 1915)
"""
    if sm_head_y_signed > 0:
        if yoff >= 0.05:
            t_detection = logger.get_timestamp_ns() if logger else 0
            head_droop = 1
            add_alert("Head Downward")
            
            # LOG MEDIAPIPE ALERT
            if logger:
                logger.log_mediapipe_alert(
                    alert_type='head_droop',
                    detection_time_ns=t_detection,
                    y_offset=yoff
                )
"""

# Looking Left/Right (around line 1860)
"""
    if gaze_left_confirm_streak >= GAZE_CONFIRM_FRAMES:
        if not gaze_alerted_left and (now_gaze - last_gaze_left_alert_time >= GAZE_ALERT_COOLDOWN):
            t_detection = logger.get_timestamp_ns() if logger else 0
            add_alert("Looking Right")
            
            # LOG MEDIAPIPE ALERT
            if logger:
                logger.log_mediapipe_alert(
                    alert_type='looking_right',
                    detection_time_ns=t_detection,
                    gaze_deviation=gaze_deviation
                )
"""

# ============================================================================
# STEP 6: Log KSS alerts (around line 2270)
# ============================================================================
"""
    # Calculate KSS score
    if logger:
        logger.mark_stage('kss', start=True)
    
    try:
        kss_features = {
            'ear_current': avg_ear,
            'head_droop_duration': head_droop_duration,
            'head_droop_active': (head_droop >= 1) and (eye_closed >= 1)
        }
        kss_score, kss_confidence = kss_calculator.calculate_kss_score(kss_features)
        
        if logger:
            logger.mark_stage('kss', start=False)
        
        # KSS override logic
        t_kss_calc = logger.get_timestamp_ns() if logger else 0
        droop_duration_direct = 0.0
        if droop_active_since is not None:
            droop_duration_direct = now_t - droop_active_since
        
        override_triggered = False
        if droop_duration_direct > 3.0:
            if avg_ear < 0.18:
                kss_score = 9.0
                kss_confidence = 0.95
                override_triggered = True
            else:
                kss_score = 8.0
                kss_confidence = 0.85
                override_triggered = True
        
        # Check for AIS 184 alerts
        if logger:
            logger.mark_stage('decision')
        
        kss_alert = kss_alert_manager.check_and_trigger_alert(kss_score, kss_confidence, now_t)
        
        if kss_alert:
            t_buzzer = logger.get_timestamp_ns() if logger else 0
            add_alert(f"[AIS 184] {kss_alert['message']} (KSS={kss_score:.1f})")
            
            if logger:
                logger.mark_stage('buzzer')
                logger.log_kss_alert(
                    kss_score=kss_score,
                    kss_confidence=kss_confidence,
                    calculation_time_ns=t_kss_calc,
                    buzzer_time_ns=t_buzzer,
                    override=override_triggered,
                    droop_duration=droop_duration_direct
                )
        
        # Console logging
        kss_label = kss_calculator.get_kss_label(kss_score)
        print(f"[KSS] Score: {kss_score:.1f}/9 - {kss_label}")
        
    except Exception as e:
        kss_score = 3.0
        kss_confidence = 0.0
"""

# ============================================================================
# STEP 7: End frame timing (at end of main loop iteration)
# ============================================================================
"""
    # END FRAME TIMING
    if logger:
        alert_occurred = (eye_closed > 0 or yawn or head_droop > 0 or kss_score >= 7.0)
        alert_type_str = None
        if eye_closed > 0:
            alert_type_str = 'eye_closed'
        elif yawn:
            alert_type_str = 'yawn'
        elif head_droop > 0:
            alert_type_str = 'head_droop'
        elif kss_score >= 7.0:
            alert_type_str = f'kss_{int(kss_score)}'
        
        logger.end_frame(alert_occurred=alert_occurred, alert_type=alert_type_str)
"""

# ============================================================================
# STEP 8: Cleanup (in finally block or at end of main())
# ============================================================================
"""
    # Cleanup
    if logger:
        latency_logger.close_logger()
"""

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
1. Copy the above snippets into your dms_integrated_mdp_yolo.py at the marked locations

2. Run with latency logging enabled:
   python dms_integrated_mdp_yolo.py --enable_latency_logs

3. Run without latency logging (normal operation):
   python dms_integrated_mdp_yolo.py

4. Analyze logs after running:
   python analyze_latency.py /tmp/dms_latency.jsonl

5. The analyzer will show:
   - End-to-end frame latencies (avg, p50, p90, p99)
   - MediaPipe alert latencies (detection to action)
   - KSS alert latencies (calculation to buzzer)
   - Stage-by-stage breakdowns
   - Worst 20 frames
   - Latency histogram

6. Logs are written to /tmp/dms_latency.jsonl (configurable with --latency_log_path)
"""
