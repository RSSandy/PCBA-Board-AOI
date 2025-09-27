from ultralytics import YOLO
import tensorflow as tf
import matplotlib.pyplot as plt


# Load an oriented bounding box (OBB) model
model = YOLO("yolov8n-obb.pt")  # You can also try yolov8s-obb.pt or larger versions

# Train
model.train(
    data="PCBA-1/data.yaml",     # Path to your data.yaml
    epochs=20,           # Adjust based on dataset size
    imgsz=640,            # Image size
    batch=16,# Lower this if GPU memory is limited
)


