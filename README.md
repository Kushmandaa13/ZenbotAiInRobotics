# ZenbotAiInRobotics

**Smart Desk-Tidying Robot (Colour Sorting)**
PDE3802 - AI in Robotics Coursework
Team: Kushmandaa, Kimberley, Leynah

---

## Overview

A colour-sorting robot system that autonomously detects desk objects using computer vision (YOLOv8), classifies them by colour, and sorts them into corresponding coloured trays using a DOFBOT 6-axis robotic arm.

## Features

- **Object Detection**: YOLOv8-based detection for 8 desk object classes
- **Colour Classification**: HSV-based colour recognition with temporal smoothing
- **Autonomous Sorting**: Pick-and-place operations with rainbow order sorting
- **Multiple Interfaces**:
  - CLI menu system (`main.py`)
  - Professional Tkinter GUI (`RobotUI.py`)
- **Performance Modes**: Fast, balanced, and accurate detection settings
- **Simulation Mode**: Test without physical hardware

## Supported Objects

| # | Object | # | Object |
|---|--------|---|--------|
| 1 | AA Battery | 5 | Highlighter |
| 2 | Charger Adapter | 6 | Pen |
| 3 | Eraser | 7 | Sharpener |
| 4 | Glue Stick | 8 | Stapler |

## Colour Trays (Rainbow Order)

Red → Orange → Yellow → Green → Blue → Indigo → Violet → Neutral

---

## Project Structure

```
ZenbotAiInRobotics/
├── src/
│   ├── main.py           # Main system orchestrator (CLI)
│   ├── vision_module.py  # Object detection & colour recognition
│   ├── arm_controller.py # DOFBOT arm control (hardcoded positions)
│   └── utils.py          # Logging, geometry, statistics utilities
├── models/
│   ├── best.pt           # YOLOv8 trained model
│   └── best_int8.tflite  # Quantized TFLite model (edge deployment)
├── RobotUI.py            # Tkinter GUI application
├── calibration.py        # Interactive servo calibration tool
├── test_arm.py           # Arm hardware tests
├── test_camera.py        # Camera/vision tests
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Hardware Requirements

- **Raspberry Pi** (tested on Pi 4/5)
- **DOFBOT 6-Axis Robotic Arm** (Yahboom)
- **USB Camera** (640×480 resolution)
- **Workspace**: 28.7cm × 30.5cm board
- **8 Coloured Trays** for sorting

## Software Requirements

- Python 3.11+
- OpenCV 4.12+
- PyTorch 2.9+
- Ultralytics (YOLOv8) 8.3+
- Tkinter (for GUI)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kushmandaa13/ZenbotAiInRobotics.git
cd ZenbotAiInRobotics
```

### 2. Create Virtual Environment

```bash
python3 -m venv robot_env
source robot_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test camera
python test_camera.py

# Test arm (simulation mode if no hardware)
python test_arm.py
```

---

## Usage

### Option 1: CLI Interface

```bash
cd src
python main.py
```

**Menu Options:**
1. Tidy Desk (Rainbow Order) - Sort by colour sequence
2. Tidy Desk (Position Order) - Sort by proximity
3. Demo Mode (Vision Only) - Live detection preview
4. Calibrate Camera - Pixel-to-cm calibration
5. Test Arm - Hardware movement tests
6. Show Statistics - Operation success rates
7. Exit

### Option 2: GUI Interface

```bash
python RobotUI.py
```

**Features:**
- Real-time video feed with detection overlay
- Confidence threshold adjustment
- Performance mode selection
- Single object or auto-sort (rainbow order)
- Operation statistics

### Option 3: Programmatic Usage

```python
from src.main import DOFBOTSystem

# Initialize system
system = DOFBOTSystem(
    model_path='models/best.pt',
    camera_id=0,
    performance_mode='balanced'
)

# Connect to arm
system.connect_arm()

# Scan and sort all objects
system.tidy_desk(rainbow_order=True)

# Cleanup
system.shutdown()
```

---

## Configuration

### Performance Modes

| Mode | Detection Size | Speed | Accuracy |
|------|---------------|-------|----------|
| fast | 256×192 | High | Lower |
| balanced | 320×256 | Medium | Medium |
| accurate | 416×320 | Lower | High |

### Arm Calibration

Use the calibration tool to adjust servo positions:

```bash
python calibration.py
```

**Controls:**
- `1-6`: Select servo
- `+/-`: Adjust ±5 degrees
- `[/]`: Fine adjust ±1 degree
- `h`: Home position
- `p`: Print current position
- `s`: Save position note

---

## Architecture

```
Camera Input
     ↓
┌─────────────────────────────────────┐
│         Vision Module               │
│  • Image Enhancement (CLAHE, WB)    │
│  • YOLOv8 Object Detection          │
│  • HSV Colour Classification        │
│  • Coordinate Transformation        │
└─────────────────────────────────────┘
     ↓
Detection Dictionary:
  {class_name, confidence, bbox,
   center, map_coord_cm, colour}
     ↓
┌─────────────────────────────────────┐
│        System Orchestrator          │
│  • Group detections by colour       │
│  • Sort by rainbow order/position   │
│  • Coordinate pick-and-place        │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│         Arm Controller              │
│  • Zone-based pickup (8 zones)      │
│  • Object-specific grip strength    │
│  • Tray placement (8 trays)         │
│  • Gripper preservation during move │
└─────────────────────────────────────┘
     ↓
Sorted Objects in Colour Trays
```

---

## Workspace Layout

```
┌─────────────────────────────────────┐
│  Zone 1  │  Zone 2  │  Zone 3  │ Z4 │  ← Row 1 (v: 0-8cm)
├──────────┼──────────┼──────────┼────┤
│  Zone 5  │  Zone 6  │  Zone 7  │ Z8 │  ← Row 2 (v: 8-16cm)
└─────────────────────────────────────┘
    0cm        7cm       14cm      21cm   28.7cm
                   (u-axis)

Trays (around workspace):
  [Red][Orange][Yellow][Green][Blue][Indigo][Violet][Neutral]
```

---

## Key Technical Details

### Gripper Settings
- Open: 60°
- Closed: 180°
- Object-specific grip values for secure handling

### Timing
- Normal move: 1500ms
- Slow move: 2000ms
- Gripper action: 800ms
- Settle time: 500ms

### Colour Detection
- HSV-based with multiple sample points
- Voting mechanism across samples
- Temporal smoothing (5-frame history)
- Black/White/Neutral detection thresholds

---

## Troubleshooting

### Camera Not Found
```bash
# List available cameras
ls /dev/video*

# Try different camera ID
python -c "import cv2; cap = cv2.VideoCapture(1); print(cap.isOpened())"
```

### Arm Not Responding
```bash
# Check if Arm_Lib is installed
python -c "from Arm_Lib import Arm_Device; print('OK')"

# Run in simulation mode (no hardware needed)
# The system auto-detects and switches to simulation
```

### Model Loading Error
```bash
# Verify model exists
ls -la models/best.pt

# Check ultralytics installation
pip install ultralytics --upgrade
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `src/main.py` | Main system orchestrator with CLI menu |
| `src/vision_module.py` | YOLOv8 detection + HSV colour classification |
| `src/arm_controller.py` | DOFBOT control with hardcoded servo positions |
| `src/utils.py` | Logging, geometry, statistics, timing utilities |
| `RobotUI.py` | Professional Tkinter GUI for robot control |
| `calibration.py` | Interactive servo position calibration |
| `test_arm.py` | Hardware test suite for arm movements |
| `test_camera.py` | Camera and vision module tests |

---

## Logs

Logs are stored in the `logs/` directory:
- `dofbot.log` - System operation log
- `operation_log.json` - JSON-formatted operation history

---

## License

This project is coursework for PDE3802 - AI in Robotics.

---

## Acknowledgements

- YOLOv8 by Ultralytics
- DOFBOT by Yahboom
- OpenCV community
