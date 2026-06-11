import cv2
from ultralytics import YOLO

# =====================================
# Load YOLO Model
# =====================================

model = YOLO("yolov8s.pt")

# =====================================
# Image Path
# =====================================

image_path = "im.png"

image = cv2.imread(image_path)

if image is None:
    print("Image not found")
    exit()

# =====================================
# Classes to Monitor
# =====================================

counts = {
    "person": 0,
    "bicycle": 0,
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}

# =====================================
# YOLO Detection
# =====================================

results = model(
    image,
    conf=0.25
)

for result in results:

    for box in result.boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        confidence = float(box.conf[0])

        if class_name not in counts:
            continue

        counts[class_name] += 1

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Bounding Box

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label

        cv2.putText(
            image,
            f"{class_name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# =====================================
# Display Counts
# =====================================

display_names = {
    "person": "Persons",
    "bicycle": "Bicycles",
    "car": "Cars",
    "motorcycle": "Motorcycles",
    "bus": "Buses",
    "truck": "Trucks"
}

y_pos = 40

for cls_name, count in counts.items():

    if count > 0:

        cv2.putText(
            image,
            f"{display_names[cls_name]}: {count}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        y_pos += 40

# =====================================
# Console Output
# =====================================

print("\nDetected Objects\n")

for cls_name, count in counts.items():

    if count > 0:

        print(
            f"{display_names[cls_name]}: {count}"
        )

# =====================================
# Full Screen Display
# =====================================

cv2.namedWindow(
    "People and Vehicle Detection",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "People and Vehicle Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

cv2.imshow(
    "People and Vehicle Detection",
    image
)

cv2.waitKey(0)

cv2.destroyAllWindows()