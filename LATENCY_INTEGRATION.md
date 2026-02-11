# DMS Latency Tracking Integration Guide

## Quick Start (3 steps)

### 1. Add import at top of `dms_integrated_mdp_yolo.py`:

```python
# Add after other imports (around line 5)
try:
    import dms_latency_tracker as lat
    _LAT = True
except:
    _LAT = False
```

### 2. Add timing markers in main loop:

```python
while cap.isOpened():
    # === STEP 1: Camera Capture ===
    if _LAT: lat.start_frame(fid)
    ret, frame = cap.read()
    if _LAT: lat.mark('t_cap')
    
    # ... preprocessing ...
    
    # === STEP 2a: MediaPipe Processing ===
    face_res = face_mesh.process(rgb)
    if _LAT: lat.mark('t_mp')
    
    # === STEP 2b: YOLO Processing ===
    yolo_worker.submit(frame)
    yolo_dets = yolo_worker.get_latest()
    if _LAT: lat.mark('t_yolo')
    
    # === STEP 3a: KSS Calculation ===
    kss_info = kss_calculator.process_frame(...)
    if _LAT: lat.mark('t_kss')
    
    # === STEP 4a: KSS Buzzer ===
    if kss_info.get('alert'):
        buzzer_beep()
        if _LAT: 
            lat.mark('t_buzzer')
            lat.log_kss_pipeline(fid)  # Logs: camera->MP->KSS->buzzer
    
    # === STEP 4b: YOLO Buzzer (seatbelt/smoking) ===
    if no_seatbelt_detected:
        buzzer_beep()
        if _LAT:
            lat.mark('t_yolo_buzzer')
            lat.log_yolo_pipeline(fid, 'no_seatbelt')  # Logs: camera->YOLO->buzzer
```

### 3. Run and see output:

```bash
python dms_integrated_mdp_yolo.py
```

**Output format:**
```
[KSS_PIPELINE] Frame 1234
  Camera Capture: 12.34 ms
  MediaPipe:      45.67 ms
  KSS Calc:       8.90 ms
  Buzzer:         2.10 ms
  ─────────────────────────────
  END-TO-END:     68.91 ms

[YOLO_PIPELINE] Frame 1235 - no_seatbelt
  Camera Capture: 11.23 ms
  YOLO Detect:    78.45 ms
  Buzzer:         1.89 ms
  ─────────────────────────────
  END-TO-END:     91.57 ms

==========================================================
LATENCY SUMMARY (every 5 seconds)
==========================================================
Camera Capture: avg=12.3ms p50=11.8ms p90=15.2ms
MediaPipe:      avg=46.2ms p50=44.1ms p90=52.3ms
YOLO:           avg=79.8ms p50=78.2ms p90=85.7ms
KSS Calc:       avg=9.1ms p50=8.8ms p90=10.2ms
Buzzer:         avg=2.0ms p50=1.9ms

END-TO-END PIPELINES:
KSS Pipeline:   avg=69.6ms p50=67.2ms p90=78.1ms (n=12)
YOLO Pipeline:  avg=93.1ms p50=91.5ms p99=102.3ms (n=8)
==========================================================
```

## Exact Locations to Add Markers

### Location 1: Camera Read (around line 1500)
```python
while cap.isOpened():
    if _LAT: lat.start_frame(fid)  # ADD THIS
    ret, frame = cap.read()
    if _LAT: lat.mark('t_cap')      # ADD THIS
```

### Location 2: MediaPipe (search for "face_mesh.process")
```python
face_res = face_mesh.process(rgb_for_mp)
if _LAT: lat.mark('t_mp')  # ADD THIS
```

### Location 3: YOLO (search for "yolo_worker.get_latest")
```python
yolo_dets = yolo_worker.get_latest() if yolo_worker else []
if _LAT: lat.mark('t_yolo')  # ADD THIS
```

### Location 4: KSS (search for "kss_calculator.process_frame")
```python
kss_info = kss_calculator.process_frame(...)
if _LAT: lat.mark('t_kss')  # ADD THIS
```

### Location 5a: KSS Buzzer (search for "buzzer_beep(times=3")
```python
if kss_info.get('alert'):
    buzzer_beep(times=3)
    if _LAT:
        lat.mark('t_buzzer')
        lat.log_kss_pipeline(fid)  # ADD BOTH LINES
```

### Location 5b: YOLO Buzzer (seatbelt - search for seatbelt buzzer)
```python
if no_seatbelt:
    buzzer_beep(times=3)
    if _LAT:
        lat.mark('t_yolo_buzzer')
        lat.log_yolo_pipeline(fid, 'no_seatbelt')  # ADD BOTH LINES
```

### Location 5c: YOLO Buzzer (smoking)
```python
if smoking:
    buzzer_beep(times=2)
    if _LAT:
        lat.mark('t_yolo_buzzer')
        lat.log_yolo_pipeline(fid, 'smoking')  # ADD BOTH LINES
```

## Understanding the Output

### Pipeline 1: KSS (Drowsiness Detection)
```
Camera → MediaPipe → KSS Logic → Buzzer
12ms   → 46ms      → 9ms       → 2ms    = 69ms total
```

### Pipeline 2: YOLO (Seatbelt/Smoking Detection)
```
Camera → YOLO → Buzzer
12ms   → 79ms → 2ms    = 93ms total
```

### Latency Breakdown:
- **Camera Capture**: Time to read frame from V4L2/GStreamer (~10-15ms normal)
- **MediaPipe**: Face mesh + hands + pose processing (~40-60ms)
- **YOLO**: Object detection inference (~70-100ms on NPU)
- **KSS**: Drowsiness score calculation (~5-10ms)
- **Buzzer**: Hardware beep activation (~1-3ms)

### Summary Statistics (every 5 seconds):
- **avg**: Average latency
- **p50**: Median (50th percentile)
- **p90**: 90th percentile (worst 10% excluded)
- **n**: Number of samples

## Files Created:
1. `dms_latency_tracker.py` - Latency tracking library (already created)
2. `LATENCY_INTEGRATION.md` - This guide

## Advanced: Using Full Logger

For detailed frame-by-frame analysis, use the full logger:

```python
from dms_latency_logger import init_logger, close_logger

# In main():
logger = init_logger('/tmp/dms_latency.jsonl', enabled=True)

# At end:
close_logger()

# Analyze:
python analyze_latency.py /tmp/dms_latency.jsonl
```

## Troubleshooting

**No output?**
- Check `_LAT` is True (print it at startup)
- Verify `dms_latency_tracker.py` is in same folder

**Wrong timings?**
- Ensure markers are in correct order
- Don't skip t_start() at frame begin
- Always call log_*_pipeline() AFTER buzzer

**Missing summaries?**
- Summaries print every 5 seconds
- Need at least 1 complete pipeline to show stats
