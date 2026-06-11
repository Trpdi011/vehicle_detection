# Vehicle Detection System

## Project Overview

This project is an AI-based Vehicle Detection System developed using YOLOv8 and OpenCV.

The system detects and classifies vehicles from images, videos, webcams, and future CCTV feeds.

Currently supported vehicle classes:

* Car
* Bus
* Truck
* Motorcycle
* Bicycle

---

## Features

### Static Image Detection

Detect vehicles from traffic images and display:

* Vehicle Class
* Bounding Boxes
* Vehicle Counts

### Video Detection

Process recorded traffic videos and display:

* Real-Time Vehicle Detection
* Vehicle Counts
* Vehicle Classification

### Webcam Detection

Detect vehicles from a live camera feed.

### ROI Monitoring

Monitor vehicles only within a selected Region of Interest (ROI).

---

## Technologies Used

* Python
* OpenCV
* YOLOv8
* Ultralytics

---

## Project Structure

```text
vehicle_detection/

├── main.py
├── pv.py
├── vehicle_detection_video.py

├── im.jpg
├── ima.jpg
├── ima1.jpg
├── ima2.jpg
├── OIP.jpg

├── obj.mp4

├── yolov8n.pt

└── README.md
```

---

## Installation

Install required packages:

```bash
pip install ultralytics
pip install opencv-python
```

---

## Model

This project uses:

```text
yolov8n.pt
```

The model file must be placed inside the project directory.

---

## Running Image Detection

Update image path inside the script:

```python
image_path = "ima2.jpg"
```

Run:

```bash
python main.py
```

---

## Running Video Detection

Update video path if required:

```python
video_path = "obj.mp4"
```

Run:

```bash
python vehicle_detection_video.py
```

---

## Output

The system displays:

* Detected Vehicles
* Bounding Boxes
* Vehicle Labels
* Vehicle Counts

Example:

```text
Car: 25
Motorcycle: 8
Bus: 3
Truck: 2
```

---

## Current Status

Completed

✔ Vehicle Detection from Images

✔ Vehicle Detection from Videos

✔ Webcam Testing

✔ Vehicle Counting

✔ ROI-Based Monitoring

In Progress

• Vehicle Database Integration

• Vehicle Analytics Reports

• Auto-Rickshaw Detection

Future Enhancements

• CCTV Integration

• Traffic Analytics Dashboard

• Historical Reports

• Custom Vehicle Model Training

---

## Known Limitations

* Detection accuracy may reduce in highly congested traffic conditions.
* Vehicles that are partially visible may occasionally be misclassified.
* Auto-Rickshaw detection requires custom training and is not included in the default YOLO model.

---

## Future Scope

* Smart Traffic Monitoring
* Vehicle Flow Analysis
* Traffic Density Monitoring
* Real-Time CCTV Deployment
* Automated Traffic Reports

