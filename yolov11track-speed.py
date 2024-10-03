import cv2
from ultralytics import YOLO
from speed import SpeedEstimator

# Load YOLOv8 model
model = YOLO("yolo11n_float32.tflite")
# Initialize global variable to store cursor coordinates
line_pts = [(0, 124), (1019, 124)]
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

speed_obj = SpeedEstimator(reg_pts=line_pts, names=class_names)

# Mouse callback function to capture mouse movement
def RGB(event, x, y, flags, param):
    global cursor_point
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_point = (x, y)
        print(f"Mouse coordinates: {cursor_point}")

# Set up the window and attach the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file or webcam feed
cap = cv2.VideoCapture('highway1.mp4')

count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be read.")
        break

    count += 1
    if count % 2 != 0:  # Skip some frames for speed (optional)
        continue

    frame = cv2.resize(frame, (1020, 500))
    
    # Perform object tracking
    tracks = model.track(frame, persist=True)
    
    
    im0 = speed_obj.estimate_speed(frame,tracks)
    
    # Display the frame with YOLOv8 results
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
