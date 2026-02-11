#!/usr/bin/env python3
"""
DMS Latency Tracker - Lightweight pipeline monitor
Tracks: Camera Capture -> MediaPipe/YOLO -> KSS -> Buzzer

Usage: Add simple timing prints to your DMS script, this analyzes them.
"""

import re
import time
from collections import deque
import statistics
from datetime import datetime


class LatencyTracker:
    def __init__(self):
        self.frame_times = {}  # frame_id -> {'t_start': ..., 't_cap': ..., }
        self.current_frame = 0
        
        # Statistics
        self.cap_latencies = deque(maxlen=1000)
        self.mp_latencies = deque(maxlen=1000)
        self.yolo_latencies = deque(maxlen=1000)
        self.kss_latencies = deque(maxlen=1000)
        self.buzzer_latencies = deque(maxlen=500)
        
        # Face recognition pipeline
        self.scrfd_latencies = deque(maxlen=500)
        self.facerec_latencies = deque(maxlen=500)
        self.facerec_e2e = deque(maxlen=500)  # camera -> scrfd -> mobilefacenet
        
        self.kss_e2e = deque(maxlen=500)  # camera -> mediapipe -> kss -> buzzer
        self.yolo_e2e = deque(maxlen=500)  # camera -> yolo -> buzzer
        
        self.last_summary_time = time.time()
        self.summary_interval = 5.0  # 5 seconds
        
        # Latency log files
        self.latency_log_file = "dms_latency_log.txt"
        self.facerec_log_file = "dms_face_recog_latency_log.txt"
        
    def mark(self, stage, frame_id=None):
        """Mark a pipeline stage completion"""
        t = time.monotonic_ns()
        
        if frame_id is None:
            frame_id = self.current_frame
        
        if frame_id not in self.frame_times:
            self.frame_times[frame_id] = {}
        
        self.frame_times[frame_id][stage] = t
        
    def start_frame(self, frame_id):
        """Start timing a new frame"""
        self.current_frame = frame_id
        self.mark('t_start', frame_id)
        
    def log_kss_pipeline(self, frame_id):
        """Log complete KSS pipeline: camera -> mediapipe -> kss -> buzzer"""
        if frame_id not in self.frame_times:
            return
        
        ft = self.frame_times[frame_id]
        
        required_keys = ['t_start', 't_cap', 't_mp', 't_kss', 't_buzzer']
        missing = [k for k in required_keys if k not in ft]
        if missing:
            return
        
        if all(k in ft for k in ['t_start', 't_cap', 't_mp', 't_kss', 't_buzzer']):
            cap_ms = (ft['t_cap'] - ft['t_start']) / 1e6
            mp_ms = (ft['t_mp'] - ft['t_cap']) / 1e6
            kss_ms = (ft['t_kss'] - ft['t_mp']) / 1e6
            buzzer_ms = (ft['t_buzzer'] - ft['t_kss']) / 1e6
            e2e_ms = (ft['t_buzzer'] - ft['t_start']) / 1e6
            
            self.cap_latencies.append(cap_ms)
            self.mp_latencies.append(mp_ms)
            self.kss_latencies.append(kss_ms)
            self.buzzer_latencies.append(buzzer_ms)
            self.kss_e2e.append(e2e_ms)
            
            # Calculate running average
            avg_e2e = statistics.mean(self.kss_e2e)
            n_count = len(self.kss_e2e)
            
            # Write to file with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            try:
                with open(self.latency_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n[{timestamp}] Eye_closure Frame {frame_id}\n")
                    f.write(f"  Data Proc: {cap_ms:.2f} ms\n")
                    f.write(f"  Algorithm: {mp_ms:.2f} ms\n")
                    f.write(f"  KSS Calc:  {kss_ms:.2f} ms\n")
                    f.write(f"  Buzzer:    {buzzer_ms:.2f} ms\n")
                    f.write(f"  ─────────────────────────────\n")
                    f.write(f"  END-TO-END: {e2e_ms:.2f} ms\n")
                    f.write(f"  KSS Pipeline: avg={avg_e2e:.1f}ms (n={n_count})\n")
                print(f"[LATENCY] Written to {self.latency_log_file}")
            except Exception as e:
                print(f"[ERROR] Failed to write latency log: {e}")
    
    def print_kss7_latency(self, frame_id, kss_score):
        """Print KSS >= 7.0 latency to TERMINAL (not file) when threshold first hit"""
        if frame_id not in self.frame_times:
            print(f"[KSS_7.0_LATENCY] No timing data for frame {frame_id}")
            return
        
        ft = self.frame_times[frame_id]
        
        required_keys = ['t_start', 't_cap', 't_mp', 't_kss', 't_buzzer']
        missing = [k for k in required_keys if k not in ft]
        if missing:
            print(f"[KSS_7.0_LATENCY] Missing timing markers: {missing}")
            return
        
        # Calculate latencies (same as eye closure logging)
        cap_ms = (ft['t_cap'] - ft['t_start']) / 1e6
        mp_ms = (ft['t_mp'] - ft['t_cap']) / 1e6
        kss_ms = (ft['t_kss'] - ft['t_mp']) / 1e6
        buzzer_ms = (ft['t_buzzer'] - ft['t_kss']) / 1e6
        e2e_ms = (ft['t_buzzer'] - ft['t_start']) / 1e6
        
        # Get running stats if available
        avg_e2e = statistics.mean(self.kss_e2e) if self.kss_e2e else e2e_ms
        n_count = len(self.kss_e2e)
        
        # Print to terminal immediately (same format as file logging)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] KSS >= 7.0 ALERT LATENCY")
        print(f"  Data Proc (Cam):   {cap_ms:7.2f} ms")
        print(f"  Algorithm (MP):    {mp_ms:7.2f} ms")
        print(f"  KSS Calculation:   {kss_ms:7.2f} ms")
        print(f"  Buzzer Trigger:    {buzzer_ms:7.2f} ms")
        print(f"  ─────────────────────────────────────")
        print(f"  END-TO-END:        {e2e_ms:7.2f} ms")
        if n_count > 0:
            print(f"  KSS Pipeline Avg:  {avg_e2e:7.2f} ms (n={n_count})")
        print(f"{'='*70}\n")
    
    def log_facerec_pipeline(self, frame_id):
        """Log face recognition pipeline: camera -> scrfd -> mobilefacenet"""
        if frame_id not in self.frame_times:
            return
        
        ft = self.frame_times[frame_id]
        
        required_keys = ['t_start', 't_cap', 't_scrfd', 't_facerec']
        missing = [k for k in required_keys if k not in ft]
        if missing:
            return
        
        if all(k in ft for k in ['t_start', 't_cap', 't_scrfd', 't_facerec']):
            cap_ms = (ft['t_cap'] - ft['t_start']) / 1e6
            scrfd_ms = (ft['t_scrfd'] - ft['t_cap']) / 1e6
            facerec_ms = (ft['t_facerec'] - ft['t_scrfd']) / 1e6
            e2e_ms = (ft['t_facerec'] - ft['t_start']) / 1e6
            
            self.cap_latencies.append(cap_ms)
            self.scrfd_latencies.append(scrfd_ms)
            self.facerec_latencies.append(facerec_ms)
            self.facerec_e2e.append(e2e_ms)
            
            # Calculate running average
            avg_e2e = statistics.mean(self.facerec_e2e)
            n_count = len(self.facerec_e2e)
            
            # Print to terminal
            print(f"\n[FACE_RECOG_LATENCY] Frame {frame_id}")
            print(f"  Data Proc: {cap_ms:.2f} ms")
            print(f"  Face D:     {scrfd_ms:.2f} ms")
            print(f"  Face V: {facerec_ms:.2f} ms")
            print(f"  ─────────────────────────────")
            print(f"  END-TO-END: {e2e_ms:.2f} ms")
            print(f"  Face Recog Pipeline: avg={avg_e2e:.1f}ms (n={n_count})\n")
    
    def log_yolo_pipeline(self, frame_id, alert_type=''):
        """Log complete YOLO pipeline: camera -> yolo -> buzzer"""
        if frame_id not in self.frame_times:
            return
        
        ft = self.frame_times[frame_id]
        if all(k in ft for k in ['t_start', 't_cap', 't_yolo', 't_yolo_buzzer']):
            cap_ms = (ft['t_cap'] - ft['t_start']) / 1e6
            yolo_ms = (ft['t_yolo'] - ft['t_cap']) / 1e6
            buzzer_ms = (ft['t_yolo_buzzer'] - ft['t_yolo']) / 1e6
            e2e_ms = (ft['t_yolo_buzzer'] - ft['t_start']) / 1e6
            
            self.cap_latencies.append(cap_ms)
            self.yolo_latencies.append(yolo_ms)
            self.buzzer_latencies.append(buzzer_ms)
            self.yolo_e2e.append(e2e_ms)
            
            print(f"\n[YOLO_PIPELINE] Frame {frame_id} - {alert_type}")
            print(f"  Camera Capture: {cap_ms:.2f} ms")
            print(f"  YOLO Detect:    {yolo_ms:.2f} ms")
            print(f"  Buzzer:         {buzzer_ms:.2f} ms")
            print(f"  ─────────────────────────────")
            print(f"  END-TO-END:     {e2e_ms:.2f} ms\n")
    
    def print_summary(self):
        """Print periodic summary"""
        if not self.cap_latencies:
            return
        
        print("\n" + "="*70)
        print("LATENCY SUMMARY")
        print("="*70)
        
        # Camera capture
        if self.cap_latencies:
            avg = statistics.mean(self.cap_latencies)
            p50 = statistics.median(self.cap_latencies)
            p90 = statistics.quantiles(self.cap_latencies, n=10)[8] if len(self.cap_latencies) >= 10 else max(self.cap_latencies)
            print(f"Camera Capture: avg={avg:.1f}ms p50={p50:.1f}ms p90={p90:.1f}ms")
        
        # MediaPipe
        if self.mp_latencies:
            avg = statistics.mean(self.mp_latencies)
            p50 = statistics.median(self.mp_latencies)
            p90 = statistics.quantiles(self.mp_latencies, n=10)[8] if len(self.mp_latencies) >= 10 else max(self.mp_latencies)
            print(f"MediaPipe:      avg={avg:.1f}ms p50={p50:.1f}ms p90={p90:.1f}ms")
        
        # YOLO
        if self.yolo_latencies:
            avg = statistics.mean(self.yolo_latencies)
            p50 = statistics.median(self.yolo_latencies)
            p90 = statistics.quantiles(self.yolo_latencies, n=10)[8] if len(self.yolo_latencies) >= 10 else max(self.yolo_latencies)
            print(f"YOLO:           avg={avg:.1f}ms p50={p50:.1f}ms p90={p90:.1f}ms")
        
        # KSS
        if self.kss_latencies:
            avg = statistics.mean(self.kss_latencies)
            p50 = statistics.median(self.kss_latencies)
            p90 = statistics.quantiles(self.kss_latencies, n=10)[8] if len(self.kss_latencies) >= 10 else max(self.kss_latencies)
            print(f"KSS Calc:       avg={avg:.1f}ms p50={p50:.1f}ms p90={p90:.1f}ms")
        
        # Buzzer
        if self.buzzer_latencies:
            avg = statistics.mean(self.buzzer_latencies)
            p50 = statistics.median(self.buzzer_latencies)
            print(f"Buzzer:         avg={avg:.1f}ms p50={p50:.1f}ms")
        
        print()
        print("END-TO-END PIPELINES:")
        
        # KSS E2E
        if self.kss_e2e:
            avg = statistics.mean(self.kss_e2e)
            p50 = statistics.median(self.kss_e2e)
            p90 = statistics.quantiles(self.kss_e2e, n=10)[8] if len(self.kss_e2e) >= 10 else max(self.kss_e2e)
            print(f"KSS Pipeline:   avg={avg:.1f}ms p50={p50:.1f}ms p90={p90:.1f}ms (n={len(self.kss_e2e)})")
        
        # YOLO E2E
        if self.yolo_e2e:
            avg = statistics.mean(self.yolo_e2e)
            p50 = statistics.median(self.yolo_e2e)
            p90 = statistics.quantiles(self.yolo_e2e, n=10)[8] if len(self.yolo_e2e) >= 10 else max(self.yolo_e2e)
            print(f"YOLO Pipeline:  avg={avg:.1f}ms p50={p50:.1f}ms p90={p90:.1f}ms (n={len(self.yolo_e2e)})")
        
        print("="*70 + "\n")
        
        self.last_summary_time = time.time()
    
    def check_summary(self):
        """Check if summary should be printed"""
        # Disabled - no periodic summary printing
        pass


# Global instance
_tracker = LatencyTracker()

def get_tracker():
    return _tracker


# Convenience functions
def start_frame(frame_id):
    _tracker.start_frame(frame_id)

def mark(stage, frame_id=None):
    _tracker.mark(stage, frame_id)

def log_kss_pipeline(frame_id):
    _tracker.log_kss_pipeline(frame_id)
    _tracker.check_summary()

def print_kss7_latency(frame_id, kss_score):
    """Print KSS >= 7.0 latency to terminal"""
    _tracker.print_kss7_latency(frame_id, kss_score)

def log_facerec_pipeline(frame_id):
    _tracker.log_facerec_pipeline(frame_id)
    # No summary check - face recog is rare event

def log_yolo_pipeline(frame_id, alert_type=''):
    _tracker.log_yolo_pipeline(frame_id, alert_type)
    _tracker.check_summary()

def end_frame():
    """End frame timing (optional, triggers summary check)"""
    _tracker.check_summary()
