"""
yolo_detection.py - YOLO Component Detection Module

PURPOSE:
Runs YOLO inference on preprocessed PCBA image to detect and classify components.
Outputs component annotations in the same format as the training dataset.

INPUTS:
- Preprocessed PCBA image
- YOLO weights file (weights.pt)

OUTPUTS:
- JSON file with component detections in Roboflow format

DEPENDENCIES:
- ultralytics (YOLOv8)
- torch
- opencv-python
- numpy

USAGE:
python3 yolo_detection.py --image preprocessed.jpg --weights ./models/weights.pt --output components.json
"""

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: pip install ultralytics torch")
    sys.exit(1)

def load_yolo_model(weights_path):
    """
    Load YOLO model from weights file
    
    Args:
        weights_path: Path to YOLO weights (.pt file)
    
    Returns:
        YOLO model instance
    """
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    
    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Print model info
    print(f"Model classes: {list(model.names.values())}")
    print(f"Model device: {model.device}")
    
    return model

def run_inference(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run YOLO inference on image
    
    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    
    Returns:
        tuple: (original_image, results)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Running inference on: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run inference
    results = model(
        image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    return image, results[0]  # Return first (and only) result

def convert_to_roboflow_format(image, results, image_path):
    """
    Convert YOLO results to Roboflow format matching training dataset
    
    Args:
        image: Original image array
        results: YOLO results object
        image_path: Path to original image
    
    Returns:
        dict: Component data in Roboflow format
    """
    height, width = image.shape[:2]
    
    # Extract detections
    boxes = results.boxes
    predictions = []
    
    if boxes is not None and len(boxes) > 0:
        # Convert tensors to numpy
        xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        conf = boxes.conf.cpu().numpy()  # Confidence scores
        cls = boxes.cls.cpu().numpy()    # Class indices
        
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            confidence = float(conf[i])
            class_id = int(cls[i])
            class_name = results.names[class_id]
            
            # Convert to center-point format (x, y, width, height)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = x1 + bbox_width / 2
            center_y = y1 + bbox_height / 2
            
            prediction = {
                "width": float(bbox_width),
                "height": float(bbox_height),
                "x": float(center_x),
                "y": float(center_y),
                "confidence": confidence,
                "class_id": class_id,
                "class": class_name,
                "detection_id": f"detection_{i:04d}",
                "parent_id": "image"
            }
            
            predictions.append(prediction)
    
    # Create Roboflow-style output
    roboflow_output = {
        "predictions_v2": {
            "image": {
                "width": width,
                "height": height
            },
            "predictions": predictions
        },
        "output_image_v2": f"<image_data_for_{Path(image_path).name}>",
        "count_objects": len(predictions)
    }
    
    return roboflow_output

def visualize_detections(image, results, output_path=None):
    """
    Create visualization of detections (optional)
    
    Args:
        image: Original image
        results: YOLO results
        output_path: Optional path to save visualization
    
    Returns:
        numpy.ndarray: Annotated image
    """
    annotated_image = image.copy()
    
    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            confidence = conf[i]
            class_id = int(cls[i])
            class_name = results.names[class_id]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_image, 
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), 
                         color, -1)
            cv2.putText(annotated_image, label, 
                       (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"Visualization saved to: {output_path}")
    
    return annotated_image

def main():
    """Main YOLO detection function"""
    parser = argparse.ArgumentParser(description="YOLO component detection for PCBA images")
    parser.add_argument("--image", required=True, help="Input preprocessed image")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (.pt file)")
    parser.add_argument("--output", required=True, help="Output JSON file for component detections")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--visualize", help="Optional path to save detection visualization")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if not Path(args.image).exists():
            raise FileNotFoundError(f"Input image not found: {args.image}")
        
        # Load YOLO model
        model = load_yolo_model(args.weights)
        
        # Run inference
        image, results = run_inference(model, args.image, args.conf, args.iou)
        
        # Convert to Roboflow format
        component_data = convert_to_roboflow_format(image, results, args.image)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(component_data, f, indent=2)
        
        # Print summary
        num_detections = len(component_data['predictions_v2']['predictions'])
        print(f"Detection complete!")
        print(f"Found {num_detections} components")
        print(f"Results saved to: {output_path}")
        
        # Component class summary
        if num_detections > 0:
            class_counts = {}
            for pred in component_data['predictions_v2']['predictions']:
                class_name = pred['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("Component counts:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count}")
        
        # Optional visualization
        if args.visualize:
            visualize_detections(image, results, args.visualize)
        
        return 0
        
    except Exception as e:
        print(f"YOLO detection failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())