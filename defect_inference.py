"""
defect_inference.py - TensorFlow Lite Defect Detection

PURPOSE:
Runs trained TFLite defect detection model on grid patches to identify defects.
Processes all patches and outputs defect predictions with confidence scores.

INPUTS:
- Directory of 224x224 patch images
- Patch metadata JSON
- Trained TFLite model

OUTPUTS:
- JSON file with defect predictions for each patch

DEPENDENCIES:
- tflite-runtime (or tensorflow-lite)
- opencv-python
- numpy
- json

USAGE:
python3 defect_inference.py --patches-dir patches/ --model pcba_defect_model.tflite --output predictions.json
"""

import json
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        TFLITE_AVAILABLE = False
        print("Using TensorFlow instead of tflite-runtime")
    except ImportError:
        print("Error: Neither tflite-runtime nor tensorflow found")
        print("Install with: pip install tflite-runtime")
        sys.exit(1)

# Default label names from training (update if different)
DEFAULT_LABEL_NAMES = [
    'has_capacitor', 'has_connector', 'has_diode', 'has_ic', 'has_inductor', 
    'has_led', 'has_potentiometer', 'has_relay', 'has_resistor', 'has_transistor',
    'has_dirt', 'has_missing', 'has_rotate', 'has_solder',
    'has_any_component', 'has_any_defect', 'is_background'
]

def load_tflite_model(model_path):
    """
    Load TensorFlow Lite model
    
    Args:
        model_path: Path to .tflite model file
    
    Returns:
        TFLite interpreter
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading TFLite model: {model_path}")
    
    if TFLITE_AVAILABLE:
        interpreter = tflite.Interpreter(model_path=str(model_path))
    else:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model input shape: {input_details[0]['shape']}")
    print(f"Model output shape: {output_details[0]['shape']}")
    
    return interpreter, input_details, output_details

def load_patch_metadata(patches_dir):
    """
    Load patch metadata from JSON file
    
    Args:
        patches_dir: Directory containing patches and metadata
    
    Returns:
        dict: Patch metadata
    """
    metadata_file = Path(patches_dir) / "patches_metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Patch metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded metadata for {metadata['total_patches']} patches")
    return metadata

def preprocess_patch(patch_image, target_size=(224, 224)):
    """
    Preprocess patch image for model inference
    
    Args:
        patch_image: Input patch image
        target_size: Target input size for model
    
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Ensure correct size
    if patch_image.shape[:2] != target_size:
        patch_image = cv2.resize(patch_image, target_size)
    
    # Convert BGR to RGB (if needed)
    if len(patch_image.shape) == 3 and patch_image.shape[2] == 3:
        patch_image = cv2.cvtColor(patch_image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    patch_array = patch_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    patch_array = np.expand_dims(patch_array, axis=0)
    
    return patch_array

def run_inference_on_patch(interpreter, input_details, output_details, patch_array):
    """
    Run inference on a single patch
    
    Args:
        interpreter: TFLite interpreter
        input_details: Model input details
        output_details: Model output details
        patch_array: Preprocessed patch array
    
    Returns:
        numpy.ndarray: Model predictions
    """
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], patch_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    return predictions[0]  # Remove batch dimension

def process_predictions(predictions, label_names=None, confidence_threshold=0.5):
    """
    Process raw model predictions into interpretable results
    
    Args:
        predictions: Raw model output
        label_names: List of label names
        confidence_threshold: Threshold for positive predictions
    
    Returns:
        dict: Processed predictions
    """
    if label_names is None:
        label_names = DEFAULT_LABEL_NAMES
    
    # Ensure we have the right number of labels
    if len(predictions) != len(label_names):
        print(f"Warning: Model output size ({len(predictions)}) doesn't match label count ({len(label_names)})")
        # Truncate or pad as needed
        min_len = min(len(predictions), len(label_names))
        predictions = predictions[:min_len]
        label_names = label_names[:min_len]
    
    # Create prediction dictionary
    prediction_dict = {}
    active_labels = []
    
    for i, (label, confidence) in enumerate(zip(label_names, predictions)):
        confidence_float = float(confidence)
        prediction_dict[label] = confidence_float
        
        if confidence_float > confidence_threshold:
            active_labels.append(label)
    
    # Determine primary classification
    max_confidence_idx = np.argmax(predictions)
    primary_label = label_names[max_confidence_idx]
    primary_confidence = float(predictions[max_confidence_idx])
    
    return {
        'all_predictions': prediction_dict,
        'active_labels': active_labels,
        'primary_label': primary_label,
        'primary_confidence': primary_confidence,
        'num_active_labels': len(active_labels)
    }

def run_inference_on_all_patches(patches_dir, interpreter, input_details, output_details, 
                                metadata, confidence_threshold=0.5):
    """
    Run inference on all patches in directory
    
    Args:
        patches_dir: Directory containing patch images
        interpreter: TFLite interpreter
        input_details: Model input details
        output_details: Model output details
        metadata: Patch metadata
        confidence_threshold: Threshold for positive predictions
    
    Returns:
        list: Results for all patches
    """
    patches_dir = Path(patches_dir)
    patch_results = []
    
    total_patches = len(metadata['patches'])
    print(f"Running inference on {total_patches} patches...")
    
    start_time = time.time()
    
    for i, patch_meta in enumerate(metadata['patches']):
        # Load patch image
        patch_file = patches_dir / patch_meta['patch_filename']
        
        if not patch_file.exists():
            print(f"Warning: Patch file not found: {patch_file}")
            continue
        
        patch_image = cv2.imread(str(patch_file))
        if patch_image is None:
            print(f"Warning: Could not load patch: {patch_file}")
            continue
        
        # Preprocess
        patch_array = preprocess_patch(patch_image)
        
        # Run inference
        raw_predictions = run_inference_on_patch(
            interpreter, input_details, output_details, patch_array
        )
        
        # Process predictions
        processed_predictions = process_predictions(
            raw_predictions, DEFAULT_LABEL_NAMES, confidence_threshold
        )
        
        # Combine with patch metadata
        result = {
            'patch_index': patch_meta['patch_index'],
            'patch_file': str(patch_file),
            'patch_coordinates': patch_meta['patch_coordinates'],
            'overlapping_components': patch_meta.get('overlapping_components', []),
            'num_components': patch_meta.get('num_components', 0),
            'predictions': processed_predictions,
            'inference_time': time.time() - start_time
        }
        
        patch_results.append(result)
        
        # Progress update
        if (i + 1) % 50 == 0 or (i + 1) == total_patches:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            print(f"Processed {i + 1}/{total_patches} patches (avg: {avg_time*1000:.1f}ms/patch)")
    
    total_time = time.time() - start_time
    print(f"Inference complete! Total time: {total_time:.2f}s")
    
    return patch_results

def analyze_results(patch_results):
    """
    Analyze inference results and create summary
    
    Args:
        patch_results: List of patch inference results
    
    Returns:
        dict: Analysis summary
    """
    total_patches = len(patch_results)
    defect_patches = []
    background_patches = []
    component_patches = []
    
    defect_types = {}
    component_types = {}
    
    for result in patch_results:
        predictions = result['predictions']
        active_labels = predictions['active_labels']
        
        # Categorize patches
        has_defect = any(label.startswith('has_') and 
                        label in ['has_dirt', 'has_missing', 'has_rotate', 'has_solder'] 
                        for label in active_labels)
        
        is_background = 'is_background' in active_labels
        has_component = 'has_any_component' in active_labels
        
        if has_defect:
            defect_patches.append(result)
            
            # Count defect types
            for label in active_labels:
                if label in ['has_dirt', 'has_missing', 'has_rotate', 'has_solder']:
                    defect_types[label] = defect_types.get(label, 0) + 1
        
        if is_background:
            background_patches.append(result)
        
        if has_component:
            component_patches.append(result)
            
            # Count component types (from overlapping_components, not predictions)
            for comp in result['overlapping_components']:
                comp_class = comp['class']
                component_types[comp_class] = component_types.get(comp_class, 0) + 1
    
    analysis = {
        'total_patches': total_patches,
        'defect_patches': len(defect_patches),
        'background_patches': len(background_patches),
        'component_patches': len(component_patches),
        'defect_percentage': len(defect_patches) / total_patches * 100 if total_patches > 0 else 0,
        'defect_types_found': defect_types,
        'component_types_detected': component_types,
        'high_confidence_defects': len([r for r in defect_patches 
                                      if r['predictions']['primary_confidence'] > 0.8])
    }
    
    return analysis

def main():
    """Main defect inference function"""
    parser = argparse.ArgumentParser(description="Run defect detection inference on patches")
    parser.add_argument("--patches-dir", required=True, help="Directory containing patch images")
    parser.add_argument("--model", required=True, help="Path to TFLite model file")
    parser.add_argument("--output", required=True, help="Output JSON file for predictions")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold for positive predictions")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if not Path(args.patches_dir).exists():
            raise FileNotFoundError(f"Patches directory not found: {args.patches_dir}")
        if not Path(args.model).exists():
            raise FileNotFoundError(f"Model file not found: {args.model}")
        
        # Load model
        interpreter, input_details, output_details = load_tflite_model(args.model)
        
        # Load patch metadata
        metadata = load_patch_metadata(args.patches_dir)
        
        # Run inference on all patches
        patch_results = run_inference_on_all_patches(
            args.patches_dir, interpreter, input_details, output_details,
            metadata, args.confidence
        )
        
        # Analyze results
        analysis = analyze_results(patch_results)
        
        # Create final results
        final_results = {
            'metadata': {
                'model_file': str(args.model),
                'patches_directory': str(args.patches_dir),
                'confidence_threshold': args.confidence,
                'total_patches_processed': len(patch_results),
                'inference_timestamp': time.time()
            },
            'analysis': analysis,
            'patch_results': patch_results
        }
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        print(f"\nDefect Detection Summary:")
        print(f"Total patches processed: {analysis['total_patches']}")
        print(f"Defect patches found: {analysis['defect_patches']} ({analysis['defect_percentage']:.1f}%)")
        print(f"High confidence defects: {analysis['high_confidence_defects']}")
        
        if analysis['defect_types_found']:
            print("Defect types detected:")
            for defect_type, count in analysis['defect_types_found'].items():
                print(f"  {defect_type}: {count}")
        
        print(f"Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Defect inference failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())