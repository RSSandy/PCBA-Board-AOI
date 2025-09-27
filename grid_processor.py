"""
grid_processor.py - Grid Patch Generation for Defect Detection

PURPOSE:
Converts YOLO component detections into systematic 224x224 grid patches for defect detection.
Uses the same grid generation approach as training pipeline.

INPUTS:
- Preprocessed PCBA image
- Component detections JSON (from YOLO)

OUTPUTS:
- Directory of 224x224 patch images
- Metadata JSON with patch locations and component info

DEPENDENCIES:
- opencv-python
- numpy
- json

USAGE:
python3 grid_processor.py --image preprocessed.jpg --components components.json --output-dir patches/
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path

def load_image_and_components(image_path, components_json_path):
    """
    Load image and component detections
    
    Args:
        image_path: Path to preprocessed image
        components_json_path: Path to YOLO component detections
    
    Returns:
        tuple: (image, component_data)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Load component detections
    with open(components_json_path, 'r') as f:
        component_data = json.load(f)
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")
    
    # Extract components from Roboflow format
    if 'predictions_v2' in component_data:
        components = component_data['predictions_v2']['predictions']
        image_info = component_data['predictions_v2']['image']
    else:
        raise ValueError("Invalid component data format")
    
    print(f"Found {len(components)} component detections")
    
    return image, components, image_info

def convert_component_coords(components):
    """
    Convert component coordinates from center-point to top-left format
    
    Args:
        components: List of component detections in Roboflow format
    
    Returns:
        list: Components with top-left bounding boxes
    """
    converted_components = []
    
    for comp in components:
        # Convert center coordinates to top-left
        x_center = comp['x']
        y_center = comp['y']
        width = comp['width']
        height = comp['height']
        
        x_topleft = x_center - width / 2
        y_topleft = y_center - height / 2
        
        converted_comp = {
            'bbox': [x_topleft, y_topleft, width, height],
            'class': comp['class'].lower(),  # Normalize to lowercase
            'confidence': comp['confidence'],
            'detection_id': comp['detection_id']
        }
        
        converted_components.append(converted_comp)
    
    return converted_components

def generate_grid_patches(image, patch_size=224, stride=112):
    """
    Generate systematic grid of patches from image
    
    Args:
        image: Input image
        patch_size: Size of square patches
        stride: Stride between patch centers
    
    Returns:
        tuple: (patches, patch_positions)
    """
    height, width = image.shape[:2]
    patches = []
    patch_positions = []
    
    # Calculate patch positions
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Ensure we don't go out of bounds
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            
            # Skip if patch is too small
            if (y_end - y) < patch_size or (x_end - x) < patch_size:
                continue
            
            # Extract patch
            patch = image[y:y_end, x:x_end]
            
            # Ensure patch is exactly patch_size x patch_size
            if patch.shape[:2] != (patch_size, patch_size):
                patch = cv2.resize(patch, (patch_size, patch_size))
            
            patches.append(patch)
            patch_positions.append({
                'x': x,
                'y': y,
                'width': patch_size,
                'height': patch_size,
                'patch_index': len(patches) - 1
            })
    
    print(f"Generated {len(patches)} patches")
    return patches, patch_positions

def calculate_component_overlaps(patch_positions, components, overlap_threshold=0.3):
    """
    Calculate which components overlap with each patch
    
    Args:
        patch_positions: List of patch position dictionaries
        components: List of component detections
        overlap_threshold: Minimum overlap percentage to consider
    
    Returns:
        list: Patch metadata with component overlaps
    """
    patch_metadata = []
    
    for patch_pos in patch_positions:
        patch_bbox = [patch_pos['x'], patch_pos['y'], patch_pos['width'], patch_pos['height']]
        
        overlapping_components = []
        component_overlaps = []
        
        # Check each component for overlap
        for comp in components:
            overlap_pct = calculate_bbox_overlap_percentage(patch_bbox, comp['bbox'])
            
            if overlap_pct > overlap_threshold:
                overlapping_components.append({
                    'class': comp['class'],
                    'confidence': comp['confidence'],
                    'detection_id': comp['detection_id'],
                    'overlap_percentage': overlap_pct
                })
                component_overlaps.append(overlap_pct)
        
        # Create patch metadata
        patch_meta = {
            'patch_index': patch_pos['patch_index'],
            'patch_coordinates': [patch_pos['x'], patch_pos['y'], 
                                patch_pos['x'] + patch_pos['width'], 
                                patch_pos['y'] + patch_pos['height']],
            'overlapping_components': overlapping_components,
            'component_overlaps': component_overlaps,
            'num_components': len(overlapping_components),
            'has_components': len(overlapping_components) > 0
        }
        
        patch_metadata.append(patch_meta)
    
    return patch_metadata

def calculate_bbox_overlap_percentage(patch_bbox, comp_bbox):
    """
    Calculate what percentage of the patch overlaps with a component
    
    Args:
        patch_bbox: [x, y, width, height] of patch
        comp_bbox: [x, y, width, height] of component
    
    Returns:
        float: Overlap percentage (0.0 to 1.0)
    """
    px1, py1, pw, ph = patch_bbox
    px2, py2 = px1 + pw, py1 + ph
    
    cx1, cy1, cw, ch = comp_bbox
    cx2, cy2 = cx1 + cw, cy1 + ch
    
    # Calculate intersection
    left = max(px1, cx1)
    top = max(py1, cy1)
    right = min(px2, cx2)
    bottom = min(py2, cy2)
    
    if left < right and top < bottom:
        intersection_area = (right - left) * (bottom - top)
        patch_area = pw * ph
        return intersection_area / patch_area if patch_area > 0 else 0
    
    return 0.0

def save_patches_and_metadata(patches, patch_metadata, output_dir, image_name):
    """
    Save patches as individual images and metadata as JSON
    
    Args:
        patches: List of patch images
        patch_metadata: List of patch metadata dictionaries
        output_dir: Output directory path
        image_name: Original image name for filename generation
    
    Returns:
        tuple: (patches_saved, metadata_file_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual patch images
    patch_files = []
    for i, patch in enumerate(patches):
        patch_filename = f"{image_name}_patch_{i:04d}.jpg"
        patch_path = output_dir / patch_filename
        
        cv2.imwrite(str(patch_path), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        patch_files.append(str(patch_path))
    
    # Update metadata with file paths
    for i, meta in enumerate(patch_metadata):
        meta['patch_file'] = patch_files[i]
        meta['patch_filename'] = Path(patch_files[i]).name
    
    # Save metadata
    metadata_file = output_dir / "patches_metadata.json"
    metadata_content = {
        'source_image': image_name,
        'total_patches': len(patches),
        'patch_size': 224,
        'stride': 112,
        'patches': patch_metadata
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_content, f, indent=2)
    
    print(f"Saved {len(patches)} patches to: {output_dir}")
    print(f"Metadata saved to: {metadata_file}")
    
    return len(patches), str(metadata_file)

def create_patch_summary(patch_metadata):
    """
    Create summary statistics of generated patches
    
    Args:
        patch_metadata: List of patch metadata
    
    Returns:
        dict: Summary statistics
    """
    total_patches = len(patch_metadata)
    patches_with_components = sum(1 for p in patch_metadata if p['has_components'])
    background_patches = total_patches - patches_with_components
    
    # Component class distribution
    component_classes = {}
    for patch in patch_metadata:
        for comp in patch['overlapping_components']:
            class_name = comp['class']
            component_classes[class_name] = component_classes.get(class_name, 0) + 1
    
    summary = {
        'total_patches': total_patches,
        'patches_with_components': patches_with_components,
        'background_patches': background_patches,
        'background_percentage': (background_patches / total_patches * 100) if total_patches > 0 else 0,
        'component_class_distribution': component_classes
    }
    
    return summary

def main():
    """Main grid processing function"""
    parser = argparse.ArgumentParser(description="Generate grid patches for defect detection")
    parser.add_argument("--image", required=True, help="Input preprocessed image")
    parser.add_argument("--components", required=True, help="Component detections JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for patches")
    parser.add_argument("--patch-size", type=int, default=224, help="Patch size (default: 224)")
    parser.add_argument("--stride", type=int, default=112, help="Patch stride (default: 112)")
    parser.add_argument("--overlap-threshold", type=float, default=0.3, 
                       help="Overlap threshold for component assignment (default: 0.3)")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if not Path(args.image).exists():
            raise FileNotFoundError(f"Input image not found: {args.image}")
        if not Path(args.components).exists():
            raise FileNotFoundError(f"Components file not found: {args.components}")
        
        # Load data
        image, components, image_info = load_image_and_components(args.image, args.components)
        
        # Convert component coordinates
        converted_components = convert_component_coords(components)
        
        # Generate grid patches
        patches, patch_positions = generate_grid_patches(
            image, args.patch_size, args.stride
        )
        
        # Calculate component overlaps
        patch_metadata = calculate_component_overlaps(
            patch_positions, converted_components, args.overlap_threshold
        )
        
        # Save patches and metadata
        image_name = Path(args.image).stem
        patches_saved, metadata_file = save_patches_and_metadata(
            patches, patch_metadata, args.output_dir, image_name
        )
        
        # Create summary
        summary = create_patch_summary(patch_metadata)
        
        print("\nPatch Generation Summary:")
        print(f"Total patches: {summary['total_patches']}")
        print(f"Patches with components: {summary['patches_with_components']}")
        print(f"Background patches: {summary['background_patches']} ({summary['background_percentage']:.1f}%)")
        
        if summary['component_class_distribution']:
            print("Component distribution:")
            for class_name, count in sorted(summary['component_class_distribution'].items()):
                print(f"  {class_name}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"Grid processing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())