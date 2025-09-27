"""
PHASE 2: DATA PARSING & CONSOLIDATION

This script takes the matched images from Phase 1 and parses all annotation data into a unified format.

Key Functions:
1. Parse component annotations from Roboflow JSON files (convert center coords to top-left)
2. Parse defect annotations from COCO format (map category IDs to names)
3. Load and validate actual image files 
4. Create consolidated data structure with all annotations per image
5. Validate that image dimensions match between annotations and actual files

Input: 
- matched_images.txt (from Phase 1)
- Component JSON files in /component_labels/
- COCO defect annotations in /defect-data/train/
- Image files in /component_images/

Output:
- consolidated_data.json (unified annotations for all matched images)
- validation_report.txt (dimension mismatches and parsing errors)
- parsed_data/ folder with individual parsed files for debugging

Data Structure Created:
{
  "IMG_3903": {
    "image_path": "path/to/IMG_3903.jpg",
    "image_dimensions": [width, height],
    "components": [
      {
        "bbox": [x, y, width, height],  # Top-left format
        "class": "LED",
        "confidence": 0.85,
        "detection_id": "..."
      }
    ],
    "defects": [
      {
        "bbox": [x, y, width, height],
        "category": "missing",
        "annotation_id": 123
      }
    ]
  }
}
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_and_consolidate_data(base_path):
    """Main function to parse and consolidate all annotation data"""
    
    base_path = Path(base_path)
    
    print("=== PHASE 2: DATA PARSING & CONSOLIDATION ===\n")
    
    # Load matched images from Phase 1
    matched_images = load_matched_images(base_path)
    print(f"1. Loading {len(matched_images)} matched images from Phase 1")
    
    # Parse component annotations
    print("\n2. Parsing component annotations...")
    component_data = parse_component_annotations(base_path, matched_images)
    
    # Parse defect annotations  
    print("\n3. Parsing defect annotations...")
    defect_data = parse_defect_annotations(base_path, matched_images)
    
    # Load and validate images
    print("\n4. Loading and validating images...")
    image_data, validation_errors = load_and_validate_images(base_path, matched_images, component_data)
    
    # Consolidate all data
    print("\n5. Consolidating data...")
    consolidated_data = consolidate_all_data(matched_images, component_data, defect_data, image_data)
    
    # Save consolidated data
    print("\n6. Saving consolidated data...")
    save_consolidated_data(base_path, consolidated_data, validation_errors)
    
    print(f"\n✅ Phase 2 Complete!")
    print(f"📊 Processed {len(consolidated_data)} images")
    print(f"📁 Check {base_path / 'consolidated_data.json'} for unified annotations")
    print(f"📁 Check {base_path / 'validation_report.txt'} for any issues")
    
    return consolidated_data

def load_matched_images(base_path):
    """Load the list of matched images from Phase 1"""
    matched_file = base_path / "matched_images.txt"
    
    if not matched_file.exists():
        raise FileNotFoundError(f"matched_images.txt not found. Run Phase 1 first.")
    
    matched_images = []
    with open(matched_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                matched_images.append(line)
    
    return matched_images

def parse_component_annotations(base_path, matched_images):
    """Parse component annotations from Roboflow JSON files"""
    
    component_labels_dir = base_path / "component_labels"
    component_data = {}
    parsing_errors = []
    
    for img_name in matched_images:
        json_file = component_labels_dir / f"{img_name}.json"
        
        if not json_file.exists():
            parsing_errors.append(f"Component JSON not found: {json_file}")
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Parse the nested Roboflow structure
            if 'predictions_v2' in data and 'predictions' in data['predictions_v2']:
                predictions = data['predictions_v2']['predictions']
                image_info = data['predictions_v2']['image']
                
                components = []
                for pred in predictions:
                    # Convert center coordinates to top-left
                    x_center = pred['x']
                    y_center = pred['y']
                    width = pred['width']
                    height = pred['height']
                    
                    # Convert to top-left format
                    x_topleft = x_center - width / 2
                    y_topleft = y_center - height / 2
                    
                    component = {
                        'bbox': [x_topleft, y_topleft, width, height],
                        'class': pred['class'],
                        'confidence': pred['confidence'],
                        'detection_id': pred['detection_id'],
                        'class_id': pred['class_id']
                    }
                    components.append(component)
                
                component_data[img_name] = {
                    'components': components,
                    'image_info': image_info,
                    'total_components': len(components)
                }
                
                print(f"   Parsed {len(components)} components from {img_name}")
                
            else:
                parsing_errors.append(f"Unexpected JSON structure in {json_file}")
                
        except Exception as e:
            parsing_errors.append(f"Error parsing {json_file}: {str(e)}")
    
    if parsing_errors:
        print(f"   ⚠️  {len(parsing_errors)} parsing errors encountered")
    
    return component_data

def parse_defect_annotations(base_path, matched_images):
    """Parse defect annotations from COCO format"""
    
    coco_file = base_path / "defect-data" / "train" / "_annotations.coco.json"
    
    if not coco_file.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_file}")
    
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"   Found defect categories: {list(categories.values())}")
    
    # Create image ID to filename mapping  
    image_id_to_name = {}
    for img in coco_data['images']:
        # Extract original name using the same logic from Phase 1
        if 'extra' in img and 'name' in img['extra']:
            original_name_with_ext = img['extra']['name']
            if '.jpg-' in original_name_with_ext:
                original_name = original_name_with_ext.split('.jpg-')[0]
            else:
                original_name = Path(original_name_with_ext).stem
            image_id_to_name[img['id']] = original_name
    
    # Group annotations by image name
    defect_data = defaultdict(list)
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        img_name = image_id_to_name.get(image_id)
        
        if img_name in matched_images:
            defect = {
                'bbox': ann['bbox'],  # Already in [x, y, width, height] format
                'category': categories[ann['category_id']],
                'category_id': ann['category_id'],
                'annotation_id': ann['id'],
                'area': ann.get('area', 0)
            }
            defect_data[img_name].append(defect)
    
    # Convert to regular dict and add summary stats
    defect_data = dict(defect_data)
    
    total_defects = 0
    for img_name in defect_data:
        count = len(defect_data[img_name])
        total_defects += count
        print(f"   Found {count} defects in {img_name}")
    
    print(f"   Total defects across all matched images: {total_defects}")
    
    return defect_data

def load_and_validate_images(base_path, matched_images, component_data):
    """Load actual image files and validate dimensions"""
    
    component_images_dir = base_path / "component_images"
    image_data = {}
    validation_errors = []
    
    for img_name in matched_images:
        img_file = component_images_dir / f"{img_name}.jpg"
        
        if not img_file.exists():
            validation_errors.append(f"Image file not found: {img_file}")
            continue
        
        try:
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                validation_errors.append(f"Could not load image: {img_file}")
                continue
            
            height, width = img.shape[:2]
            
            # Validate dimensions against component annotations
            if img_name in component_data:
                expected_width = component_data[img_name]['image_info']['width']
                expected_height = component_data[img_name]['image_info']['height']
                
                if width != expected_width or height != expected_height:
                    validation_errors.append(
                        f"Dimension mismatch for {img_name}: "
                        f"actual ({width}x{height}) vs expected ({expected_width}x{expected_height})"
                    )
            
            image_data[img_name] = {
                'path': str(img_file),
                'dimensions': [width, height],
                'channels': img.shape[2] if len(img.shape) > 2 else 1
            }
            
            print(f"   Loaded {img_name}: {width}x{height}")
            
        except Exception as e:
            validation_errors.append(f"Error loading {img_file}: {str(e)}")
    
    if validation_errors:
        print(f"   ⚠️  {len(validation_errors)} validation errors encountered")
    
    return image_data, validation_errors

def consolidate_all_data(matched_images, component_data, defect_data, image_data):
    """Combine all parsed data into unified structure"""
    
    consolidated = {}
    
    for img_name in matched_images:
        if img_name not in image_data:
            continue  # Skip images that failed to load
        
        consolidated[img_name] = {
            'image_path': image_data[img_name]['path'],
            'image_dimensions': image_data[img_name]['dimensions'],
            'components': component_data.get(img_name, {}).get('components', []),
            'defects': defect_data.get(img_name, []),
            'total_components': len(component_data.get(img_name, {}).get('components', [])),
            'total_defects': len(defect_data.get(img_name, []))
        }
    
    return consolidated

def save_consolidated_data(base_path, consolidated_data, validation_errors):
    """Save consolidated data and validation report"""
    
    # Create output directory
    parsed_dir = base_path / "parsed_data"
    parsed_dir.mkdir(exist_ok=True)
    
    # Save consolidated data
    consolidated_file = base_path / "consolidated_data.json"
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    
    print(f"   📁 Saved consolidated data to {consolidated_file}")
    
    # Save individual parsed files for debugging
    for img_name, data in consolidated_data.items():
        individual_file = parsed_dir / f"{img_name}_parsed.json"
        with open(individual_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"   📁 Saved individual parsed files to {parsed_dir}")
    
    # Save validation report
    validation_file = base_path / "validation_report.txt"
    with open(validation_file, 'w') as f:
        f.write("# PHASE 2 VALIDATION REPORT\n")
        f.write(f"# Generated during data parsing and consolidation\n\n")
        
        f.write(f"## SUMMARY\n")
        f.write(f"Total images processed: {len(consolidated_data)}\n")
        f.write(f"Total validation errors: {len(validation_errors)}\n\n")
        
        if validation_errors:
            f.write(f"## VALIDATION ERRORS\n")
            for error in validation_errors:
                f.write(f"- {error}\n")
        else:
            f.write(f"## NO VALIDATION ERRORS FOUND\n")
        
        f.write(f"\n## DATA STATISTICS\n")
        total_components = sum(data['total_components'] for data in consolidated_data.values())
        total_defects = sum(data['total_defects'] for data in consolidated_data.values())
        f.write(f"Total components across all images: {total_components}\n")
        f.write(f"Total defects across all images: {total_defects}\n")
        f.write(f"Average components per image: {total_components / len(consolidated_data):.1f}\n")
        f.write(f"Average defects per image: {total_defects / len(consolidated_data):.1f}\n")
    
    print(f"   📁 Saved validation report to {validation_file}")

# Main execution
if __name__ == "__main__":
    BASE_PATH = "/Users/sandhyanayar/Development/PCBAHackathon/defect_model_data"
    
    consolidated_data = parse_and_consolidate_data(BASE_PATH)