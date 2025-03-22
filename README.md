# Face Align

A real-time face tracking and auto-zooming application that smoothly keeps faces centered in frame using YOLOv8. The application provides smooth transitions and intelligent tracking to keep faces well-framed during video calls or recordings.

## Features

- Real-time face detection using YOLOv8
- Smooth camera transitions with configurable parameters
- Automatic zoom and center on detected faces
- Smart movement thresholding to prevent jitter
- Efficient frame processing with reduced detection frequency
- Maintains consistent aspect ratio

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


The application will open your webcam feed and automatically start tracking and center your face.

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
