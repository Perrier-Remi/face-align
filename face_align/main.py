import cv2
from ultralytics import YOLO
from face_align.zoomedImage import ZoomedImage
from pathlib import Path

def process_frame_with_detection(frame, model, zoomed_image, last_box):
    # No need for classes parameter as YOLOv8-face only detects faces
    results = model(frame, conf=0.5)
    result = results[0]
    
    if len(result.boxes) > 0:
        # Get the face with highest confidence
        box = result.boxes[0].xyxy[0]
        last_box = box
    else:
        last_box = None  # Reset last_box when no face is detected

    zoomed_frame = zoomed_image.process_frame(frame, last_box)
    return zoomed_frame, last_box, box if len(result.boxes) > 0 else None

def main():
    # Setup model paths and URLs
    model_path = Path('models/yolov8n-face.pt')
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
    current_box = None
    show_rectangle = False  # Toggle for showing face rectangle

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        # Reduce detection frequency
        if frame_count % 3 == 0:
            zoomed_frame, last_box, current_box = process_frame_with_detection(
                frame, model, zoomed_image, last_box)
            frame_count = 0
        frame_count += 1

        zoomed_frame = zoomed_image.process_frame(frame, last_box)
        
        # Draw rectangle if enabled and we have a current detection
        if show_rectangle and current_box is not None:
            x1, y1, x2, y2 = map(int, current_box)
            cv2.rectangle(zoomed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(zoomed_frame, 'Face', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Align', zoomed_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 'q' or Esc to quit
            break
        elif key == ord('r'):  # 'r' to toggle rectangle
            show_rectangle = not show_rectangle

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

