# Face Align

A real-time face tracking and auto-zooming application that smoothly keeps faces centered in frame using YOLOv8. The application provides smooth transitions and intelligent tracking to keep faces well-framed during video calls or recordings. Inspired by Apple's Center Stage feature found in modern MacBooks and iPads.

## Features

- Real-time face detection using YOLOv8
- Smooth camera transitions with configurable parameters
- Automatic zoom and center on detected faces (similar to Apple's Center Stage)
- Smart movement thresholding to prevent jitter

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/face-align.git
cd face-align
```

2. Install Poetry if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:

```bash
poetry install
```

## Usage

Run the application:

```bash
poetry run face_align
```

The application will open your webcam feed and automatically start tracking and center your face, similar to how Center Stage works on Apple devices.

Controls:
- Press 'q' or 'Esc' to quit

## Configuration

The tracking behavior can be customized by adjusting parameters in `ZoomedImage`:

- `margin_percent`: Extra space around the detected face (default: 12)
- `smoothing_factor`: Lower values give smoother transitions (default: 0.07)
- `threshold`: Minimum pixel movement before updating position (default: 35)

## Requirements

- Python 3.10+
- Poetry for dependency management
- Webcam

All Python dependencies are managed through Poetry and specified in `pyproject.toml`.
