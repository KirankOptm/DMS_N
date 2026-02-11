"""
KSS (Karolinska Sleepiness Scale) Calculator for AIS 184 Compliance
Implements drowsiness scoring system for Driver Monitoring System

KSS Scale (1-9):
1 = Extremely alert
2 = Very alert
3 = Alert
4 = Rather alert
5 = Neither alert nor sleepy
6 = Some signs of sleepiness
7 = Sleepy, no effort to stay awake (AIS 184 Alert Threshold)
8 = Sleepy, some effort to stay awake (AIS 184 Critical Threshold)
9 = Very sleepy, fighting sleep

AIS 184 Requirements:
- Alert at KSS >= 7
- Sensitivity >= 40% at KSS >= 8
- False positive rate < 20% at KSS < 7
"""

import time
from collections import deque
import numpy as np


class KSSCalculator:
    """
    Calculates KSS score (1-9) based on multiple drowsiness indicators
    """
    
    def __init__(self, perclos_window=60, blink_window=60):
        """
        Initialize KSS calculator
        
        Args:
            perclos_window: Time window in seconds for PERCLOS calculation (default 60s)
            blink_window: Time window in seconds for blink rate calculation (default 60s)
        """
        # Time windows for metrics
        self.perclos_window = perclos_window
        self.blink_window = blink_window
        
        # Rolling data storage (time-based filtering, not frame-based)
        # Conservative maxlen for low FPS systems (1-2 FPS): 60s * 3 FPS = 180 samples max
        self.ear_history = deque(maxlen=int(perclos_window * 3))  # Assume 3 FPS max (works for 1-2 FPS)
        self.ear_timestamps = deque(maxlen=int(perclos_window * 3))
        
        self.blink_history = deque(maxlen=100)  # Last 100 blinks
        self.blink_timestamps = deque(maxlen=100)
        
        self.yawn_history = deque(maxlen=50)  # Last 50 yawns
        self.yawn_timestamps = deque(maxlen=50)
        
        # Thresholds for EAR-based metrics
        self.EAR_CLOSED_THRESHOLD = 0.18  # Eyes considered closed
        self.EAR_DROWSY_THRESHOLD = 0.20  # Mild drowsiness
        
        # Configurable weights for multi-factor model
        self.weights = {
            'perclos': 0.35,      # PERCLOS_80 (most important)
            'ear_avg': 0.25,      # Average EAR
            'blink_rate': 0.15,   # Blinks per minute
            'blink_duration': 0.10,  # Average blink duration
            'yawn_rate': 0.10,    # Yawns per minute
            'head_droop': 0.05    # Head droop duration
        }
        
        # Initialize state
        self.current_kss = 3.0  # Start at "Alert"
        self.confidence = 0.0
        self.last_update = time.perf_counter()
        
        # Smoothing
        self.kss_smooth_factor = 0.3  # EMA smoothing
        
    def add_ear_sample(self, ear_value, timestamp=None):
        """
        Add EAR sample for PERCLOS calculation
        
        Args:
            ear_value: Current Eye Aspect Ratio
            timestamp: Optional timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.perf_counter()
            
        self.ear_history.append(ear_value)
        self.ear_timestamps.append(timestamp)
        
    def add_blink(self, duration, timestamp=None):
        """
        Record a blink event
        
        Args:
            duration: Blink duration in seconds
            timestamp: Optional timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.perf_counter()
            
        self.blink_history.append(duration)
        self.blink_timestamps.append(timestamp)
        
    def add_yawn(self, timestamp=None):
        """
        Record a yawn event
        
        Args:
            timestamp: Optional timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.perf_counter()
            
        self.yawn_history.append(timestamp)
        self.yawn_timestamps.append(timestamp)
        
    def calculate_perclos_80(self):
        """
        Calculate PERCLOS (Percentage of Eye Closure)
        PERCLOS_80: Percentage of time eyes are >= 80% closed
        
        Returns:
            float: PERCLOS percentage (0-100)
        """
        if len(self.ear_history) < 10:
            return 0.0
            
        current_time = time.perf_counter()
        
        # Filter samples within time window
        valid_samples = []
        for ear, ts in zip(self.ear_history, self.ear_timestamps):
            if current_time - ts <= self.perclos_window:
                valid_samples.append(ear)
                
        if not valid_samples:
            return 0.0
            
        # Count samples below threshold (eyes closed)
        closed_count = sum(1 for ear in valid_samples if ear < self.EAR_CLOSED_THRESHOLD)
        perclos = (closed_count / len(valid_samples)) * 100.0
        
        return perclos
        
    def calculate_blink_rate(self):
        """
        Calculate blink rate (blinks per minute) in recent window
        
        Returns:
            float: Blinks per minute
        """
        if len(self.blink_timestamps) < 2:
            return 15.0  # Normal baseline
            
        current_time = time.perf_counter()
        
        # Count blinks in last window
        recent_blinks = sum(1 for ts in self.blink_timestamps 
                          if current_time - ts <= self.blink_window)
        
        # Calculate rate per minute
        time_span = min(self.blink_window, current_time - self.blink_timestamps[0])
        if time_span > 0:
            blink_rate = (recent_blinks / time_span) * 60.0
        else:
            blink_rate = 15.0
            
        return blink_rate
        
    def calculate_avg_blink_duration(self):
        """
        Calculate average blink duration in recent window
        
        Returns:
            float: Average blink duration in seconds
        """
        if len(self.blink_history) < 3:
            return 0.15  # Normal baseline
            
        current_time = time.perf_counter()
        
        # Get recent blink durations
        recent_durations = []
        for dur, ts in zip(self.blink_history, self.blink_timestamps):
            if current_time - ts <= self.blink_window:
                recent_durations.append(dur)
                
        if not recent_durations:
            return 0.15
            
        return np.mean(recent_durations)
        
    def calculate_yawn_rate(self):
        """
        Calculate yawn rate (yawns per minute) in recent window
        
        Returns:
            float: Yawns per minute
        """
        if len(self.yawn_timestamps) < 1:
            return 0.0
            
        current_time = time.perf_counter()
        
        # Count yawns in last 5 minutes (yawns are rarer)
        window = 300  # 5 minutes
        recent_yawns = sum(1 for ts in self.yawn_timestamps 
                         if current_time - ts <= window)
        
        # Calculate rate per minute
        time_span = min(window, current_time - self.yawn_timestamps[0])
        if time_span > 0:
            yawn_rate = (recent_yawns / time_span) * 60.0
        else:
            yawn_rate = 0.0
            
        return yawn_rate
        
    def calculate_kss_score(self, features):
        """
        Calculate KSS score (1-9) based on current features
        
        Args:
            features: Dictionary with keys:
                - ear_current: Current EAR value
                - avg_ear: Average EAR in recent window (optional)
                - blink_rate: Current blink rate (optional, will calculate if not provided)
                - blink_duration: Current blink duration (optional)
                - yawn_count: Number of yawns (optional, will calculate if not provided)
                - head_droop_duration: Seconds of head droop (optional)
                - head_droop_active: Boolean if head droop currently active (optional)
                
        Returns:
            tuple: (kss_score, confidence)
                kss_score: Float 1.0-9.0
                confidence: Float 0.0-1.0 indicating data quality
        """
        # Add current EAR to history
        if 'ear_current' in features:
            self.add_ear_sample(features['ear_current'])
            
        # Calculate PERCLOS
        perclos = self.calculate_perclos_80()
        
        # Get or calculate metrics
        avg_ear = features.get('avg_ear', np.mean(list(self.ear_history)) if self.ear_history else 0.25)
        blink_rate = features.get('blink_rate', self.calculate_blink_rate())
        blink_duration = features.get('blink_duration', self.calculate_avg_blink_duration())
        yawn_rate = features.get('yawn_rate', self.calculate_yawn_rate())
        head_droop_duration = features.get('head_droop_duration', 0.0)
        
        # Normalize metrics to 0-1 scale for scoring
        # Higher values = more drowsy
        
        # PERCLOS: 0-50% mapped to 0-1
        perclos_score = min(perclos / 50.0, 1.0)
        
        # Average EAR: 0.12-0.28 mapped to 1-0 (inverted, lower EAR = more drowsy)
        ear_score = max(0.0, min(1.0, (0.28 - avg_ear) / 0.16))
        
        # Blink rate: 5-25 bpm, optimal at 15, abnormal <8 or >22
        if blink_rate < 8:
            blink_score = (15 - blink_rate) / 15.0  # Slow blinking = drowsy
        elif blink_rate > 22:
            blink_score = (blink_rate - 15) / 15.0  # Rapid blinking = drowsy
        else:
            blink_score = 0.0  # Normal range
        blink_score = max(0.0, min(1.0, blink_score))
        
        # Blink duration: 0.1-0.5s, longer = more drowsy
        duration_score = max(0.0, min(1.0, (blink_duration - 0.1) / 0.4))
        
        # Yawn rate: 0-3 yawns/min mapped to 0-1
        yawn_score = min(yawn_rate / 3.0, 1.0)
        
        # Head droop: 0-10 seconds mapped to 0-1
        droop_score = min(head_droop_duration / 10.0, 1.0)
        
        # Weighted combination
        combined_score = (
            self.weights['perclos'] * perclos_score +
            self.weights['ear_avg'] * ear_score +
            self.weights['blink_rate'] * blink_score +
            self.weights['blink_duration'] * duration_score +
            self.weights['yawn_rate'] * yawn_score +
            self.weights['head_droop'] * droop_score
        )
        
        # Map combined score (0-1) to KSS scale (1-9)
        # Using piecewise linear mapping for better discrimination
        if combined_score < 0.1:
            kss_raw = 1.0 + (combined_score / 0.1) * 2.0  # 1-3: Alert
        elif combined_score < 0.3:
            kss_raw = 3.0 + ((combined_score - 0.1) / 0.2) * 2.0  # 3-5: Mild
        elif combined_score < 0.6:
            kss_raw = 5.0 + ((combined_score - 0.3) / 0.3) * 2.0  # 5-7: Moderate
        else:
            kss_raw = 7.0 + ((combined_score - 0.6) / 0.4) * 2.0  # 7-9: Severe
            
        kss_raw = max(1.0, min(9.0, kss_raw))
        
        # Apply temporal smoothing
        if self.current_kss > 0:
            kss_smoothed = (self.kss_smooth_factor * kss_raw + 
                          (1 - self.kss_smooth_factor) * self.current_kss)
        else:
            kss_smoothed = kss_raw
            
        self.current_kss = kss_smoothed
        
        # Calculate confidence based on data availability
        data_points = len(self.ear_history)
        if data_points < 30:
            self.confidence = data_points / 30.0  # Low confidence initially
        elif data_points < 100:
            self.confidence = 0.5 + (data_points - 30) / 140.0  # Building confidence
        else:
            self.confidence = 1.0  # Full confidence
            
        self.last_update = time.perf_counter()
        
        return self.current_kss, self.confidence
        
    def get_kss_label(self, kss_score):
        """
        Get descriptive label for KSS score
        
        Args:
            kss_score: KSS value (1-9)
            
        Returns:
            str: Descriptive label
        """
        if kss_score < 3:
            return "Extreme Alert"
        elif kss_score < 5:
            return "Alert"
        elif kss_score < 7:
            return "Neither Alert/Nor Sleepy"
        elif kss_score < 8:
            return "Sleepy"
        else:
            return "Extreme Drowsiness (Mandatory warning)"
            
    def reset(self):
        """Reset all history and state"""
        self.ear_history.clear()
        self.ear_timestamps.clear()
        self.blink_history.clear()
        self.blink_timestamps.clear()
        self.yawn_history.clear()
        self.yawn_timestamps.clear()
        self.current_kss = 3.0
        self.confidence = 0.0


class AIS184AlertManager:
    """
    Manages alerts according to AIS 184 standard
    """
    
    def __init__(self, buzzer_callback=None):
        """
        Initialize alert manager
        
        Args:
            buzzer_callback: Function to call for buzzer alerts (optional)
                            Should accept: buzzer_callback(times, on_s, off_s)
        """
        self.buzzer_callback = buzzer_callback
        
        # AIS 184 thresholds
        self.KSS_ALERT_THRESHOLD = 7.0      # Warning alert
        self.KSS_CRITICAL_THRESHOLD = 8.0   # Critical alert
        self.KSS_URGENT_THRESHOLD = 8.5     # Urgent alert
        
        # Alert state tracking
        self.last_alert_time = 0
        self.alert_cooldown = 10.0  # 10 seconds between alerts
        self.current_alert_level = 0  # 0=none, 1=warning, 2=critical, 3=urgent
        
        # Alert history for validation
        self.alert_log = []
        
    def check_and_trigger_alert(self, kss_score, confidence, timestamp=None):
        """
        Check KSS score and trigger appropriate alert
        
        Args:
            kss_score: Current KSS score (1-9)
            confidence: Confidence level (0-1)
            timestamp: Optional timestamp
            
        Returns:
            dict: Alert information or None
        """
        if timestamp is None:
            timestamp = time.perf_counter()
            
        # Don't alert if confidence is too low
        if confidence < 0.3:
            return None
            
        # Check cooldown
        if timestamp - self.last_alert_time < self.alert_cooldown:
            return None
            
        alert_info = None
        
        # Determine alert level
        if kss_score >= self.KSS_URGENT_THRESHOLD:
            alert_level = 3
            alert_msg = "URGENT: Driver Fighting Sleep!"
            buzzer_pattern = (8, 0.08, 0.08)  # Continuous alert: 8 fast pulses, max energy
        elif kss_score >= self.KSS_CRITICAL_THRESHOLD:
            alert_level = 2
            alert_msg = "CRITICAL: Severe Drowsiness Detected"
            buzzer_pattern = (6, 0.08, 0.08)  # Continuous alert: 6 fast pulses, max energy
        elif kss_score >= self.KSS_ALERT_THRESHOLD:
            alert_level = 1
            alert_msg = "WARNING: Driver Drowsiness Detected"
            buzzer_pattern = (1, 0.2, 0.0)  # Single long beep
        else:
            alert_level = 0
            alert_msg = None
            buzzer_pattern = None
            
        # Only alert if level increased or first time at this level
        if alert_level > 0:
            alert_info = {
                'level': alert_level,
                'message': alert_msg,
                'kss_score': kss_score,
                'confidence': confidence,
                'timestamp': timestamp
            }
            
            # Trigger buzzer
            if self.buzzer_callback and buzzer_pattern:
                try:
                    self.buzzer_callback(*buzzer_pattern)
                except Exception as e:
                    print(f"Buzzer callback error: {e}")
                    
            # Log alert
            self.alert_log.append(alert_info)
            self.last_alert_time = timestamp
            self.current_alert_level = alert_level
            
        return alert_info
        
    def get_alert_color(self, kss_score):
        """
        Get color for visual display based on KSS score
        
        Args:
            kss_score: KSS value (1-9)
            
        Returns:
            tuple: BGR color tuple for OpenCV
        """
        if kss_score < 7.0:
            return (0, 255, 0)  # Green - OK
        elif kss_score < 8.0:
            return (0, 255, 255)  # Yellow - Warning
        elif kss_score < 8.5:
            return (0, 165, 255)  # Orange - Critical
        else:
            return (0, 0, 255)  # Red - Urgent
            
    def get_statistics(self):
        """
        Get alert statistics for validation
        
        Returns:
            dict: Statistics including alert counts by level
        """
        if not self.alert_log:
            return {
                'total_alerts': 0,
                'warning_count': 0,
                'critical_count': 0,
                'urgent_count': 0
            }
            
        total = len(self.alert_log)
        warning = sum(1 for a in self.alert_log if a['level'] == 1)
        critical = sum(1 for a in self.alert_log if a['level'] == 2)
        urgent = sum(1 for a in self.alert_log if a['level'] == 3)
        
        return {
            'total_alerts': total,
            'warning_count': warning,
            'critical_count': critical,
            'urgent_count': urgent,
            'alert_log': self.alert_log
        }


# Convenience function for easy integration
def create_kss_system(buzzer_callback=None):
    """
    Create and initialize KSS calculator and alert manager
    
    Args:
        buzzer_callback: Function for buzzer control (optional)
        
    Returns:
        tuple: (kss_calculator, alert_manager)
    """
    calculator = KSSCalculator(perclos_window=60, blink_window=60)
    manager = AIS184AlertManager(buzzer_callback=buzzer_callback)
    
    return calculator, manager
