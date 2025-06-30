import cv2
import requests
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import pyttsx3
import os

# Load model
model = YOLO("best.pt")
names = model.names

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Make folder
os.makedirs("detections", exist_ok=True)

# ESP32-CAM capture URL
esp32_url = "http://192.168.43.71/capture"  # Replace with your ESP32 IP

while True:
    input("Press ENTER to capture image...")

    response = requests.get(esp32_url)
    if response.status_code != 200:
        print("Failed to get image from ESP32-CAM.")
        continue

    # Decode JPEG to OpenCV format
    img_array = np.frombuffer(response.content, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model(frame)[0]
    detection_flag = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf < 0.5:
            continue

        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Calculate label position to keep it inside the frame
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Default: above the box
        text_x = x1
        text_y = y1 - 10

        # If too close to top, show below
        if text_y < text_size[1]:
            text_y = y1 + text_size[1] + 10

        # If too close to bottom, move label above
        if text_y + text_size[1] > frame.shape[0]:
            text_y = y1 - 10

        # If text is going off the right edge
        if text_x + text_size[0] > frame.shape[1]:
            text_x = frame.shape[1] - text_size[0] - 10

        # Draw label background for better visibility
        cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2),
                    (text_x + text_size[0] + 2, text_y + 2), (0, 255, 0), -1)

        # Draw text
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)


        if label in ['weed', 'chilli']:
            detection_flag = True
            engine.say(f"{label} detected with confidence {int(conf * 100)} percent")
            engine.runAndWait()

    # Save detection if needed
    if detection_flag:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detections/detect_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

    # Show result
    cv2.imshow("Detection Result", frame)
    cv2.waitKey(1)
