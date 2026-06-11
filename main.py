import cv2
from ultralytics import YOLO

# =====================================
# Load Model
# =====================================

model = YOLO("yolov8m.pt")

# =====================================
# Load Image
# =====================================

image_path = "im.png"

image = cv2.imread(image_path)

if image is None:

    print("ERROR: Image not found")

    exit()

# =====================================
# Select ROI
# =====================================

roi = cv2.selectROI(
    "Select Road Area",
    image,
    False
)

cv2.destroyWindow(
    "Select Road Area"
)

x, y, w, h = roi

# ROI Crop

roi_image = image[
    y:y+h,
    x:x+w
]

# =====================================
# Detection
# =====================================

results = model(

    roi_image,

    imgsz=1280,

    conf=0.25,

    verbose=False

)

# =====================================
# Vehicle Classes
# =====================================

vehicle_counts = {

    "bicycle": 0,

    "car": 0,

    "motorcycle": 0,

    "bus": 0,

    "truck": 0,

    "train": 0,

    "airplane": 0

}

# =====================================
# Process Detections
# =====================================

for result in results:

    for box in result.boxes:

        confidence = float(
            box.conf[0]
        )

        if confidence < 0.25:

            continue

        cls_id = int(
            box.cls[0]
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

            # Convert ROI coordinates
            # back to original image

            x1 += x
            x2 += x

            y1 += y
            y2 += y

            cv2.rectangle(

                image,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )

            cv2.putText(

                image,

                f"{class_name} "
                f"{confidence:.2f}",

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 0),

                2

            )

# =====================================
# Draw ROI Area
# =====================================

cv2.rectangle(

    image,

    (x, y),

    (x + w, y + h),

    (255, 0, 0),

    3

)

cv2.putText(

    image,

    "ROI",

    (x, y - 10),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.8,

    (255, 0, 0),

    2

)

# =====================================
# Display Counts
# =====================================

y_pos = 35

for vehicle, count in vehicle_counts.items():

    if count > 0:

        cv2.putText(

            image,

            f"{vehicle.capitalize()}: {count}",

            (20, y_pos),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2

        )

        y_pos += 35

# =====================================
# Console Output
# =====================================

print("\nDetected Vehicles\n")

for vehicle, count in vehicle_counts.items():

    if count > 0:

        print(

            f"{vehicle}: {count}"

        )

# =====================================
# Full Screen
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

cv2.imshow(

    "Vehicle Detection",

    image

)

cv2.waitKey(0)

cv2.destroyAllWindows()