from ultralytics import YOLO
import cv2
import math

# Load YOLO model
model = YOLO("./runs/detect/train3/weights/best.pt")

# Open video file or capture from camera (change the video file path or camera index accordingly)
video_path = "C:/Users/lvish/anaconda3/envs/crack-model/_final_code/data/val/images/vid.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the current frame
    results = model(frame, stream=True, conf=0.5)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0]) * 100) / 100
            cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 225, 0), 4)
            cv2.putText(frame, f'Confidence: {conf}', (x1 + 33, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (196, 184, 6), 1)

    # Display the result
    cv2.imshow("Result", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
