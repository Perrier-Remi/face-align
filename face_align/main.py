import cv2
from ultralytics import YOLO
from face_align.zoomedImage import ZoomedImage
import os
import requests
from pathlib import Path

def download_model(url, model_path):
    """Download the model if it doesn't exist."""
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        response = requests.get(url)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")

def process_frame_with_detection(frame, model, zoomed_image, last_box):
    # No need for classes parameter as YOLOv8-face only detects faces
    results = model(frame, conf=0.5)
    result = results[0]
    
    if len(result.boxes) > 0:
        # Get the face with highest confidence
        box = result.boxes[0].xyxy[0]
        last_box = box

    zoomed_frame = zoomed_image.process_frame(frame, last_box)
    return zoomed_frame, last_box

def main():
    # Setup model paths and URLs
    model_path = Path('models/yolov8n.pt')
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    
    # Ensure model directory exists and download model if needed
    download_model(model_url, model_path)
    model = YOLO(model_path)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get actual frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_size = (frame_width, frame_height)
    frame_shape = (frame_height, frame_width, 3)  # OpenCV uses (height, width, channels)
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Initialize zoomed image processor with frame shape
    zoomed_image = ZoomedImage(target_size, frame_shape=frame_shape)
    
    frame_count = 0
    last_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        # Reduce detection frequency
        if frame_count % 5 == 0:
            zoomed_frame, last_box = process_frame_with_detection(
                frame, model, zoomed_image, last_box)
            frame_count = 0
        frame_count += 1

        zoomed_frame = zoomed_image.process_frame(frame, last_box)
        
        cv2.imshow('Face Align', zoomed_frame)

        # Press 'q' or 'Esc' to quit
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

