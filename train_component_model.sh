#!/bin/bash
# train_yolo.sh
set -e

echo "Starting YOLO training at $(date)"
echo "Working directory: $(pwd)"

# Create a temporary venv on the worker
python3 -m venv yolo-env
source yolo-env/bin/activate

# Install ultralytics inside the job environment
pip install --upgrade pip
pip install ultralytics

# Run YOLO training
yolo detect train model=$MODEL data=$DATASET_DIR/data.yaml epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH

echo "Training complete at $(date)"
