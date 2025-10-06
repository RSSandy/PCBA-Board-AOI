# Automated Optical Inspection (AOI) Pipeline for PCBAs
# Overview

This project develops a low-cost Automated Optical Inspection (AOI) system for detecting defects and missing components on printed circuit board assemblies (PCBAs). Built using a Raspberry Pi and Arduino camera, the system was designed as part of the College of Engineering’s Makerthon ’25 competition for Plexus Corp, where it earned 2nd place.

## Features

Component Identification: YOLO-based model trained to identify PCB components with 85% accuracy.

Defect Detection: Lightweight MobileNet model trained to detect defects and anomalies within identified components.

Edge Deployment: Models deployed on a Raspberry Pi for real-time inspection without external computing resources.

Synthetic Data Generation: Custom data augmentation pipeline created to compensate for limited labeled datasets.

## Challenges

No defect dataset: Created synthetic and augmented data to train the defect detection model.

Blueprint-blind design: The system was built to operate without prior knowledge of the board layout, which made identifying missing components difficult.

Limited compute resources: Training vision models without GPU access was time-intensive; Roboflow and Google Colab were used for training and experimentation.

## Technologies Used

Python, PyTorch

YOLOv8, MobileNet

OpenCV, NumPy

Raspberry Pi, Arduino Camera

## Results

Achieved 85% accuracy in component identification.

Demonstrated a scalable, cost-effective AI-based inspection solution suitable for real-world manufacturing environments.
