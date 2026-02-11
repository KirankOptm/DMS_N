# DMS Driver Authentication Integration Guide

## Overview
The DMS system now includes **face authentication** at startup. When enabled, the system will:
1. **Authenticate the driver** using SCRFD face detector + face recognition (NPU-accelerated)
2. **Show driver name** on screen if authenticated
3. **Proceed automatically** after 10 seconds if unauthorized or no face detected
4. **Continue with normal DMS monitoring** (MediaPipe + YOLO) after authentication

## Quick Start

### 1. Enroll Drivers First
Before authentication, you must enroll drivers into the database:

```bash
# Run enrollment on IMX board
python3 enroll_industrial_board.py
```

This creates `drivers.json` with enrolled driver embeddings.

### 2. Run DMS with Authentication

#### Enable Authentication (Recommended):
```bash
python3 dms_integrated_mdp_yolo.py --enable_auth
```

#### Run without Authentication:
```bash
python3 dms_integrated_mdp_yolo.py
```

## Authentication Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_auth` | False | Enable driver authentication at startup |
| `--auth_timeout` | 10 | Authentication timeout in seconds |
| `--auth_detector` | scrfd_500m_full_int8_vela.tflite | SCRFD detector model |
| `--auth_recognizer` | fr_int8_velaS.tflite | Face recognition model (NPU) |
| `--auth_database` | drivers.json | Driver database path |

## Authentication Flow

```
START DMS
    ↓
[AUTHENTICATION ENABLED?]
    ↓ YES                    ↓ NO
[Run Authentication]         [Skip to Monitoring]
    ↓
[Show Camera Feed]
    ↓
[Detect Face] ─→ [No Face for 10s] ─→ [Timeout] ─→ [Unauthorized Mode]
    ↓
[Match Face]
    ↓ Match Found          ↓ No Match
[AUTHENTICATED]            [Continue Trying]
    ↓                          ↓
[Show Driver Name]         [Timeout after 10s]
    ↓                          ↓
[Start DMS Monitoring] ←───────┘
    ↓
[MediaPipe + YOLO]
```

## Example Commands

### Full Production Setup:
```bash
# With authentication + optimized for IMX board
python3 dms_integrated_mdp_yolo.py \
  --enable_auth \
  --auth_timeout 10 \
  --fast_preset_imx8 \
  --use_gstreamer True
```

### Quick Test (No Authentication):
```bash
# Skip authentication for testing
python3 dms_integrated_mdp_yolo.py
```

### Custom Authentication Models:
```bash
# Use different models
python3 dms_integrated_mdp_yolo.py \
  --enable_auth \
  --auth_detector scrfd_500m_full_int8_vela.tflite \
  --auth_recognizer fr_int8_velaS.tflite \
  --auth_database my_drivers.json
```

## Display Information

### When Authenticated:
- **Top-left corner**: `Driver: [Name] (ID: [ID])` in GREEN
- **Normal alerts**: Display below driver name
- **Full DMS monitoring**: All features active

### When Unauthorized:
- **No driver name** displayed
- **Full DMS monitoring**: All features still active
- **Console message**: "Proceeding in unauthorized mode"

## Authentication States

### ✓ Success
```
✓ AUTHENTICATED: John Doe (ID: 1)
  Similarity: 87.5%
  Time: 2.3s
```

### ⚠ Timeout (10 seconds)
```
[Timeout] No authorized driver detected in 10s
Proceeding with monitoring in unauthorized mode...
```

### ⚠ No Enrolled Drivers
```
⚠ No enrolled drivers - skipping authentication
```

## Troubleshooting

### Authentication Always Times Out
**Problem**: Can't detect/recognize driver
**Solutions**:
1. Check lighting conditions (face must be well-lit)
2. Ensure driver looks at camera
3. Verify enrollment was successful: `ls -lh drivers.json`
4. Test detector separately: `python3 authenticate_face_board.py`

### Models Not Found
**Problem**: `FileNotFoundError: scrfd_500m_full_int8_vela.tflite`
**Solutions**:
1. Check model files exist in current directory
2. Use absolute paths: `--auth_detector /path/to/model.tflite`
3. Verify NPU delegate: `ls /usr/lib/libethosu_delegate.so`

### NPU Not Working
**Problem**: Falls back to CPU (slow)
**Solutions**:
1. Check Ethos-U delegate: `/usr/lib/libethosu_delegate.so`
2. Verify device: `ls /dev/ethosu0`
3. Use vela-optimized models (suffix: `_vela.tflite`)

### Camera Issues
**Problem**: `Cannot open camera`
**Solutions**:
1. Check camera device: `ls /dev/video*`
2. Specify camera: `--camera_device /dev/video0`
3. Test camera: `v4l2-ctl --list-devices`

## Integration Architecture

```
authenticate_face_board.py (Module)
    ├── FaceAuthenticatorBoard (Class)
    │   ├── SCRFD Detector (NPU)
    │   ├── Face Recognizer (NPU)
    │   └── Database Loader
    └── quick_authenticate() (Function)
        ├── 10-second timeout
        ├── Real-time face matching
        └── Returns: (success, name, id)

dms_integrated_mdp_yolo.py (Main)
    ├── Import: quick_authenticate
    ├── Args: --enable_auth, --auth_timeout
    ├── [Startup] Authentication Phase
    │   └── quick_authenticate() → (success, name, id)
    └── [Main Loop] DMS Monitoring
        ├── MediaPipe (face mesh, hands, pose)
        ├── YOLO (seatbelt, cigarette detection)
        └── Display: Driver name + alerts
```

## Performance Notes

- **Authentication time**: Typically 1-3 seconds for enrolled drivers
- **Timeout overhead**: Maximum 10 seconds (adjustable with `--auth_timeout`)
- **NPU acceleration**: Both SCRFD and FR models run on Ethos-U NPU
- **Zero impact**: Authentication runs once at startup, no ongoing performance cost

## Files Modified

1. **authenticate_face_board.py**
   - Added `quick_authenticate()` function for DMS integration
   - Modular design: can be imported or run standalone

2. **dms_integrated_mdp_yolo.py**
   - Imported authentication module
   - Added CLI arguments for authentication
   - Integrated authentication at startup
   - Display authenticated driver name on screen

## Next Steps

1. **Enroll your drivers**: `python3 enroll_industrial_board.py`
2. **Test authentication**: `python3 authenticate_face_board.py`
3. **Run integrated DMS**: `python3 dms_integrated_mdp_yolo.py --enable_auth`
4. **Production deployment**: Add `--fast_preset_imx8` for optimal performance

## Notes

- Authentication is **optional** - system works without it
- Unauthorized drivers still get **full DMS monitoring**
- Timeout ensures system **never blocks** indefinitely
- Driver name displays **throughout the session** once authenticated
