#!/usr/bin/env python3
"""
Test latency tracker - simulate DMS pipeline
"""

import time
import dms_latency_tracker as lat

def simulate_camera_read():
    """Simulate camera frame capture"""
    time.sleep(0.012)  # 12ms

def simulate_mediapipe():
    """Simulate MediaPipe processing"""
    time.sleep(0.046)  # 46ms

def simulate_yolo():
    """Simulate YOLO inference"""
    time.sleep(0.079)  # 79ms

def simulate_kss():
    """Simulate KSS calculation"""
    time.sleep(0.009)  # 9ms

def simulate_buzzer():
    """Simulate buzzer activation"""
    time.sleep(0.002)  # 2ms

print("Testing DMS Latency Tracker")
print("Simulating 10 frames with KSS and YOLO alerts...\n")

for frame_id in range(1, 11):
    # === KSS Pipeline (every 3 frames) ===
    if frame_id % 3 == 0:
        lat.start_frame(frame_id)
        
        simulate_camera_read()
        lat.mark('t_cap')
        
        simulate_mediapipe()
        lat.mark('t_mp')
        
        simulate_kss()
        lat.mark('t_kss')
        
        # Trigger alert
        print(f"Frame {frame_id}: KSS Alert detected!")
        simulate_buzzer()
        lat.mark('t_buzzer')
        
        lat.log_kss_pipeline(frame_id)
    
    # === YOLO Pipeline (every 5 frames) ===
    elif frame_id % 5 == 0:
        lat.start_frame(frame_id)
        
        simulate_camera_read()
        lat.mark('t_cap')
        
        simulate_yolo()
        lat.mark('t_yolo')
        
        # Trigger alert
        print(f"Frame {frame_id}: YOLO Alert (no seatbelt)!")
        simulate_buzzer()
        lat.mark('t_yolo_buzzer')
        
        lat.log_yolo_pipeline(frame_id, 'no_seatbelt')
    
    time.sleep(0.033)  # 30 FPS

print("\n✓ Test complete!")
print("Summary should appear above after 5 seconds\n")
