import cv2
import time

from datetime import datetime

from pymongo import MongoClient

from ultralytics import YOLO


# =====================================
# Load YOLO Model
# =====================================

model = YOLO("yolov8m.pt")


# =====================================
# MongoDB Connection
# =====================================

client = MongoClient(
    "mongodb://localhost:27018/"
)

db = client["vehicle_monitoring"]

vehicle_collection = db["vehicle_counts"]


# =====================================
# Open Video
# =====================================

cap = cv2.VideoCapture(
    "obj.mp4"
)

if not cap.isOpened():

    print(
        "Cannot Open Video"
    )

    exit()


# =====================================
# Full Screen Window
# =====================================

cv2.namedWindow(

    "Vehicle Detection",

    cv2.WINDOW_NORMAL

)

cv2.setWindowProperty(

    "Vehicle Detection",

    cv2.WND_PROP_FULLSCREEN,

    cv2.WINDOW_FULLSCREEN

)


# =====================================
# Save Timer
# =====================================

last_save_time = time.time()


# =====================================
# Main Loop
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Video Finished"
        )

        break

    # =====================================
    # Resize Frame
    # =====================================

    frame = cv2.resize(

        frame,

        (1280, 720)

    )

    # =====================================
    # Vehicle Counts
    # =====================================

    vehicle_counts = {

        "car": 0,

        "motorcycle": 0,

        "bus": 0,

        "truck": 0,

        "bicycle": 0

    }

    # =====================================
    # Detection
    # =====================================

    results = model(

        frame,

        conf=0.50,

        imgsz=640,

        verbose=False

    )

    # =====================================
    # Process Results
    # =====================================

    for result in results:

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[
                cls_id
            ]

            if class_name in vehicle_counts:

                vehicle_counts[
                    class_name
                ] += 1

                x1, y1, x2, y2 = map(

                    int,

                    box.xyxy[0]

                )

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2

                )

                cv2.putText(

                    frame,

                    class_name,

                    (x1, y1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2

                )

    # =====================================
    # Display Counts
    # =====================================

    y_pos = 40

    for vehicle, count in vehicle_counts.items():

        if count > 0:

            cv2.putText(

                frame,

                f"{vehicle.capitalize()}: {count}",

                (20, y_pos),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 255),

                3

            )

            y_pos += 45

    # =====================================
    # Save To MongoDB
    # Every 60 Seconds
    # =====================================

    current_time = time.time()

    if current_time - last_save_time >= 60:

        vehicle_collection.insert_one({

            "timestamp": datetime.now(),

            "cars": vehicle_counts["car"],

            "motorcycles": vehicle_counts["motorcycle"],

            "buses": vehicle_counts["bus"],

            "trucks": vehicle_counts["truck"],

            "bicycles": vehicle_counts["bicycle"]

        })

        print(

            "Vehicle Data Saved"

        )

        last_save_time = current_time

    # =====================================
    # Show Frame
    # =====================================

    cv2.imshow(

        "Vehicle Detection",

        frame

    )

    key = cv2.waitKey(1)

    # ESC Key

    if key == 27:

        print(
            "Program Closed"
        )

        break


# =====================================
# Cleanup
# =====================================

cap.release()

cv2.destroyAllWindows()