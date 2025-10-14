# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zoom 강의 출석 자동화 v2.0 - A dual-monitor desktop application that monitors Zoom video feeds in real-time, detects faces using OpenCV DNN, and automatically captures screenshots during class periods for attendance tracking.

**Key Technologies**: Python, PyQt5, OpenCV DNN, mss (screen capture), APScheduler

## Common Commands

### Running the Application

```bash
# GUI version (main application)
python desktop_app.py

# Console version for testing
python main.py --test

# Test face detection only
python face_detector.py

# Test screen capture
python screen_capture.py
```

### Building Distribution

```bash
# Build standalone executable with PyInstaller
python build.py

# The build process will:
# 1. Clean previous builds
# 2. Create icon assets
# 3. Generate/update .spec file
# 4. Build executable to dist/
# 5. Create distribution package in release/
```

### Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

### Module Structure

The codebase follows a modular architecture with clear separation of concerns:

**Core Modules:**
- `desktop_app.py` - PyQt5 GUI application (main entry point)
- `main.py` - Console-based version with integrated system
- `face_detector.py` - Memory-efficient face detection using OpenCV DNN
- `screen_capture.py` - Multi-monitor screen capture with thread-safety
- `scheduler.py` - APScheduler-based class period scheduling
- `zoom_detector.py` - Zoom participant box detection and analysis
- `monitor_selector.py` - Dual monitor management
- `notification_system.py` - System notifications and sound alerts
- `logger.py` - CSV-based attendance logging

### Memory Management Strategy

The face detection system uses **smart memory management** to reduce RAM usage by ~73%:

1. **Lazy Loading**: OpenCV DNN model only loads during detection windows (class minutes 35-50)
2. **Scheduled Detection**: Active for 15 seconds every 60 seconds during detection window
3. **Automatic Unloading**: Model unloads after detection period with forced garbage collection

**Key Implementation** in `face_detector.py`:
- `_load_model()` / `_unload_model()` - Dynamic model lifecycle
- `is_detection_time()` - Returns True only during minutes 35-50 of class periods
- `should_activate_detection()` - Controls 15s/60s duty cycle
- `start_detection_cycle()` - Schedules automatic model unloading

### Class Schedule System

**8 Class Periods** defined in `scheduler.py`:
- 1교시: 09:30-10:30
- 2교시: 10:30-11:30
- 3교시: 11:30-12:30
- 4교시: 12:30-13:30
- **점심시간: 13:30-14:30** (no capture)
- 5교시: 14:30-15:30
- 6교시: 15:30-16:30
- 7교시: 16:30-17:30
- 8교시: 17:30-18:30

**Auto-capture Window**: Minutes 35-40 of each period (5-minute window)

The scheduler uses APScheduler with cron triggers to execute capture callbacks at specific times.

### Threading Model

**Multi-threaded Architecture** in `desktop_app.py`:

1. **Main Thread**: PyQt5 event loop and UI updates
2. **CaptureThread** (QThread): Real-time screen capture and face analysis
   - Runs at 5-second intervals by default
   - Emits signals: `frame_ready`, `original_frame_ready`, `analysis_ready`
3. **Scheduler Thread**: APScheduler blocking scheduler for timed captures
4. **Detection Thread**: Timer-based model unloading (in face_detector)

**Thread Safety**:
- Screen capture uses thread-local mss instances to prevent Windows GDI `srcdc` errors
- Face detection uses threading.Lock for model loading/unloading
- PyQt signals/slots for thread-safe UI updates

### Screen Capture Architecture

`screen_capture.py` implements **thread-safe dual monitor support**:

- Uses thread-local storage (`threading.local`) to maintain separate mss instances per thread
- Prevents Windows GDI "srcdc" object errors when multiple threads capture simultaneously
- Supports monitor switching without restart
- `cleanup()` method properly releases GDI resources

**Key Pattern**:
```python
def _get_sct(self):
    if not hasattr(self._local, 'sct') or self._local.sct is None:
        self._local.sct = mss.mss()
    return self._local.sct
```

### Face Detection Models

**Primary**: YuNet (OpenCV 4.5.4+)
- Model: `face_detection_yunet_2023mar.onnx` (~2.8MB)
- Auto-downloads from opencv_zoo GitHub on first run
- Stored in `models/` directory
- **No TensorFlow required** - pure OpenCV
- **Windows-compatible** - no DLL issues

**Detection Process**:
1. Dynamic input size: YuNet adapts to actual image dimensions
2. YuNet detect() API: Returns faces with 5 facial landmarks
3. Filter by confidence threshold (default 0.6, optimized for YuNet)
4. Extract landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
5. Validate bounding boxes within image bounds

**Advantages over previous Caffe model**:
- Higher accuracy (especially for side profiles and partially occluded faces)
- Facial landmarks included (5 keypoints)
- Faster inference speed
- Better Windows compatibility
- More recent model (2023)

### GUI Architecture (desktop_app.py)

**Tab-based Interface** with QTabWidget:
- **메인 모니터링 Tab**: Real-time preview and status dashboard
- **설정 Tab**: Monitor selection, detection parameters, class schedules, logs

**Key Components**:
- `CaptureThread`: Background capture and analysis worker
- Real-time timers: `status_timer` (1s), `preview_timer` (200ms)
- QSettings persistence for user preferences
- System tray integration for background operation

**Signal Flow**:
```
CaptureThread.run()
  → screen capture
  → face detection
  → emit signals
  → UI update slots
```

### Data Persistence

**Attendance Logging** (`logger.py`):
- CSV format: date, class_period, capture_count, file_names, timestamp, status
- Excel export capability for analysis
- Stored in `attendance_log.csv`

**Image Storage** (`captures/` directory):
- Filename format: `YYYYMMDD_{period}교시_{n}.png`
- Max 5 captures per class period
- Original frames saved (no visualization overlay)

**Settings Storage** (QSettings):
- Application: 'ZoomAttendance'
- Organization: 'Settings'
- Persists: monitor choice, face count threshold, manual duration, class schedules

## Important Implementation Details

### Force Detection Mode

For testing/debugging, the system supports **force detection** that bypasses time restrictions:

```python
# In face_detector.py
detector.detect_faces(image, force_detection=True)

# In desktop_app.py CaptureThread
self.test_mode_active = True  # Enables continuous detection
```

This is used by the GUI's test mode button and manual detection timer.

### Zoom Participant Detection

`zoom_detector.py` uses **computer vision to find individual participant boxes**:

1. Convert to grayscale and apply Canny edge detection
2. Find contours representing participant boxes
3. Filter by aspect ratio and minimum size
4. Extract individual participant regions
5. Run face detection on each region
6. Visualize with color-coded boxes (green = face detected, red = no face)

### Error Handling Patterns

The codebase consistently uses try-except with logging:

```python
try:
    # operation
    self.logger.info("Success message")
except Exception as e:
    self.logger.error(f"Error context: {e}")
    return default_value
```

All modules use Python's `logging` module with consistent format.

### Windows Threading Issues

**Problem**: Windows GDI objects (srcdc) cannot be shared across threads
**Solution**: Thread-local mss instances in `screen_capture.py`

Always call `cleanup()` when stopping capture threads to release GDI resources.

## Build System

`build.py` is a comprehensive build script that:

1. Verifies dependencies (PyInstaller, OpenCV, mss, PyQt5)
2. Cleans dist/ and build/ directories
3. Creates icon assets using PIL (camera icon)
4. Generates/updates PyInstaller .spec file
5. Builds single-file executable
6. Creates distribution package with README and run scripts
7. Generates ZIP archive in release/

**Output**: `dist/ZoomAttendance.exe` (Windows) or `dist/ZoomAttendance` (Mac/Linux)

## Testing Strategies

### Component Testing

Each module has `if __name__ == "__main__":` test code:
- `face_detector.py`: Webcam test with force detection ('f' key)
- `screen_capture.py`: Single capture and candidate manager test
- `scheduler.py`: Schedule validation and time checks
- `main.py --test`: End-to-end capture and face detection

### GUI Testing

Use the GUI's built-in test features:
- **테스트 캡쳐 버튼**: Single frame capture with face detection
- **테스트 모드**: Continuous face detection (bypasses time restrictions)
- **수동 탐지**: Timed detection for specified duration

### Integration Testing

Run the full system with `python desktop_app.py`:
1. Start monitoring to verify screen capture
2. Enable scheduler to test timed captures
3. Check logs in settings tab for errors
4. Verify captures saved in `captures/` directory

## Configuration Notes

### Model Files

First run downloads ~2.8MB YuNet model to `models/`:
- `face_detection_yunet_2023mar.onnx` (~2.8MB)

If download fails, the system logs an error. Manually download from:
- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

### Monitor Numbers

mss library indexes monitors starting from 1:
- Monitor 0: Virtual combined desktop (all monitors)
- Monitor 1: Primary display
- Monitor 2: Secondary display (default for Zoom)

Check available monitors with `MonitorManager.list_all_monitors()` in `monitor_selector.py`.

## Performance Considerations

- **Memory**: With smart scheduling, baseline ~150MB, peak ~400MB during detection
- **CPU**: ~5-10% during idle monitoring, ~30-40% during face detection
- **Capture Rate**: 5 seconds per frame (configurable via `CaptureThread.set_capture_interval()`)
- **Detection Duty Cycle**: 15s active / 45s idle during capture windows

To optimize performance:
- Increase `capture_interval` for slower systems
- Reduce face detection confidence threshold if missing faces
- Disable unused class periods in settings

## Key Dependencies and Versions

From `requirements.txt`:
- `opencv-python>=4.8.0` - Computer vision and DNN
- `mss>=9.0.1` - Fast screen capture
- `APScheduler>=3.10.4` - Task scheduling
- `PyQt5>=5.15.0` - GUI framework
- `pandas>=2.0.0` - Data logging
- `numpy>=1.24.0` - Array operations
- `pyinstaller>=5.13.0` - Executable building

All dependencies are pure Python or have binary wheels for Windows/Mac/Linux.
