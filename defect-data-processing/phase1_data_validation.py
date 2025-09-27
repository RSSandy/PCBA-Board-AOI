import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import shutil

def explore_and_validate_datasets(base_path):
    """
    Phase 1: Complete data validation and exploration
    """
    base_path = Path(base_path)
    
    # Define paths
    component_labels_dir = base_path / "component_labels"
    component_images_dir = base_path / "component_images" 
    defect_data_dir = base_path / "defect-data"
    defect_train_dir = defect_data_dir / "train"
    defect_annotations_file = defect_train_dir / "_annotations.coco.json"
    
    # Create output directories
    mismatched_dir = base_path / "mismatched_data"
    mismatched_dir.mkdir(exist_ok=True)
    
    print("=== PHASE 1: DATA VALIDATION & EXPLORATION ===\n")
    
    # Step 1: Explore component dataset
    print("1. EXPLORING COMPONENT DATASET...")
    component_exploration = explore_component_dataset(component_labels_dir, component_images_dir)
    
    # Step 2: Explore defect dataset  
    print("\n2. EXPLORING DEFECT DATASET...")
    defect_exploration = explore_defect_dataset(defect_annotations_file, defect_train_dir)
    
    # Step 3: Cross-dataset matching
    print("\n3. CROSS-DATASET MATCHING...")
    matching_results = match_datasets(component_exploration, defect_exploration)
    
    # Step 4: Generate mismatch reports and move files
    print("\n4. GENERATING MISMATCH REPORTS...")
    generate_mismatch_reports(base_path, matching_results, mismatched_dir)
    
    # Step 5: Create final matched dataset list
    print("\n5. CREATING FINAL MATCHED DATASET...")
    matched_images = create_matched_dataset_list(base_path, matching_results)
    
    # Step 6: Generate exploration summary
    print("\n6. GENERATING EXPLORATION SUMMARY...")
    generate_exploration_summary(base_path, component_exploration, defect_exploration, matching_results)
    
    print(f"\n✅ Phase 1 Complete!")
    print(f"📊 Found {len(matched_images)} fully matched images")
    print(f"📁 Check {base_path / 'matched_images.txt'} for final dataset")
    print(f"📁 Check {base_path / 'exploration_summary.txt'} for detailed analysis")
    
    return matched_images, component_exploration, defect_exploration

def explore_component_dataset(labels_dir, images_dir):
    """Explore component labels and images"""
    exploration = {
        'total_label_files': 0,
        'total_image_files': 0,
        'component_classes': Counter(),
        'confidence_stats': [],
        'image_dimensions': [],
        'label_files': set(),
        'image_files': set(),
        'sample_annotations': [],
        'errors': []
    }
    
    # Explore label files
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.json"))
        exploration['total_label_files'] = len(label_files)
        
        print(f"   Found {len(label_files)} JSON label files")
        
        for i, label_file in enumerate(label_files):
            base_name = label_file.stem  # e.g., "IMG_3901" or "IMG_3901.jpg_output_image_v2"
            
            # Clean up component filenames - remove .jpg_output_image_v2 suffix if present
            if base_name.endswith('.jpg_output_image_v2'):
                base_name = base_name.replace('.jpg_output_image_v2', '')
            
            exploration['label_files'].add(base_name)
            
            try:
                with open(label_file, 'r') as f:
                    data = json.load(f)
                
                # Parse the nested structure
                if 'predictions_v2' in data and 'predictions' in data['predictions_v2']:
                    predictions = data['predictions_v2']['predictions']
                    image_info = data['predictions_v2']['image']
                    
                    # Store sample for inspection
                    if i < 3:  # Keep first 3 samples
                        exploration['sample_annotations'].append({
                            'file': str(label_file),
                            'image_dims': image_info,
                            'num_predictions': len(predictions),
                            'sample_predictions': predictions[:2]  # First 2 predictions
                        })
                    
                    # Collect statistics
                    exploration['image_dimensions'].append((image_info['width'], image_info['height']))
                    
                    for pred in predictions:
                        exploration['component_classes'][pred['class']] += 1
                        exploration['confidence_stats'].append(pred['confidence'])
                        
                else:
                    exploration['errors'].append(f"Unexpected JSON structure in {label_file}")
                    
            except Exception as e:
                exploration['errors'].append(f"Error reading {label_file}: {str(e)}")
    else:
        exploration['errors'].append(f"Component labels directory not found: {labels_dir}")
    
    # Explore image files
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg"))
        exploration['total_image_files'] = len(image_files)
        
        print(f"   Found {len(image_files)} JPG image files")
        
        for img_file in image_files:
            base_name = img_file.stem  # e.g., "IMG_3901" or "IMG_3901.jpg_output_image_v2"
            
            # Clean up component filenames - remove .jpg_output_image_v2 suffix if present
            if base_name.endswith('.jpg_output_image_v2'):
                base_name = base_name.replace('.jpg_output_image_v2', '')
            
            exploration['image_files'].add(base_name)
    else:
        exploration['errors'].append(f"Component images directory not found: {images_dir}")
    
    # Calculate statistics
    if exploration['confidence_stats']:
        confidences = exploration['confidence_stats']
        exploration['confidence_min'] = min(confidences)
        exploration['confidence_max'] = max(confidences)
        exploration['confidence_avg'] = sum(confidences) / len(confidences)
    
    return exploration

def explore_defect_dataset(annotations_file, defect_images_dir):
    """Explore defect annotations and images"""
    exploration = {
        'total_annotations': 0,
        'total_images_in_coco': 0,
        'total_image_files': 0,
        'defect_categories': {},
        'defect_counts': Counter(),
        'original_to_transformed': {},
        'transformed_to_original': {},
        'image_files': set(),
        'annotation_samples': [],
        'errors': []
    }
    
    # Parse COCO annotations
    if annotations_file.exists():
        try:
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            
            # Explore categories
            if 'categories' in coco_data:
                for cat in coco_data['categories']:
                    exploration['defect_categories'][cat['id']] = cat['name']
                print(f"   Found defect categories: {list(exploration['defect_categories'].values())}")
            
            # Explore images
            if 'images' in coco_data:
                exploration['total_images_in_coco'] = len(coco_data['images'])
                print(f"   Found {exploration['total_images_in_coco']} images in COCO annotations")
                
                for img in coco_data['images']:
                    transformed_name = img['file_name']
                    
                    # Extract original filename from extra.name if available
                    if 'extra' in img and 'name' in img['extra']:
                        original_name_with_ext = img['extra']['name']  # e.g., "IMG_3916.jpg" or "IMG_3915.jpg-713951-1758986281286-0.jpg"
                        
                        # Check if extra.name still has the complex format
                        if '.jpg-' in original_name_with_ext:
                            # Still needs extraction: IMG_3915.jpg-713951-1758986281286-0.jpg -> IMG_3915
                            original_name = original_name_with_ext.split('.jpg-')[0]
                        else:
                            # Clean format: IMG_3916.jpg -> IMG_3916
                            original_name = Path(original_name_with_ext).stem
                        
                        exploration['original_to_transformed'][original_name] = transformed_name
                        exploration['transformed_to_original'][transformed_name] = original_name
                        print(f"   Mapped: {original_name_with_ext} -> {original_name}")  # Debug line
                    else:
                        # Fallback: extract from transformed name pattern
                        # e.g., IMG_3910.jpg-695551-1758986283039-0_jpg.rf.xxx.jpg -> IMG_3910
                        if '.jpg-' in transformed_name:
                            original_name = transformed_name.split('.jpg-')[0]
                            exploration['original_to_transformed'][original_name] = transformed_name
                            exploration['transformed_to_original'][transformed_name] = original_name
                        else:
                            exploration['errors'].append(f"Could not extract original name from: {transformed_name}")
            
            # Explore annotations
            if 'annotations' in coco_data:
                exploration['total_annotations'] = len(coco_data['annotations'])
                print(f"   Found {exploration['total_annotations']} defect annotations")
                
                # Sample first few annotations
                for i, ann in enumerate(coco_data['annotations'][:5]):
                    exploration['annotation_samples'].append({
                        'category_id': ann['category_id'],
                        'category_name': exploration['defect_categories'].get(ann['category_id'], 'unknown'),
                        'bbox': ann['bbox'],
                        'image_id': ann['image_id']
                    })
                
                # Count defect types
                for ann in coco_data['annotations']:
                    cat_name = exploration['defect_categories'].get(ann['category_id'], 'unknown')
                    exploration['defect_counts'][cat_name] += 1
                    
        except Exception as e:
            exploration['errors'].append(f"Error reading COCO annotations: {str(e)}")
    else:
        exploration['errors'].append(f"COCO annotations file not found: {annotations_file}")
    
    # Explore defect image files
    if defect_images_dir.exists():
        image_files = list(defect_images_dir.glob("*.jpg"))
        exploration['total_image_files'] = len(image_files)
        
        print(f"   Found {len(image_files)} defect image files")
        
        for img_file in image_files:
            file_name = img_file.name
            exploration['image_files'].add(file_name)
    else:
        exploration['errors'].append(f"Defect images directory not found: {defect_images_dir}")
    
    return exploration

def match_datasets(component_exploration, defect_exploration):
    """Cross-reference datasets to find matches and mismatches"""
    
    # Get sets of base filenames
    component_labels = component_exploration['label_files']
    component_images = component_exploration['image_files'] 
    defect_originals = set(defect_exploration['original_to_transformed'].keys())
    
    print(f"   Component labels: {len(component_labels)} files")
    print(f"   Component images: {len(component_images)} files") 
    print(f"   Defect annotations: {len(defect_originals)} original names")
    
    # Find intersections and differences
    all_names = component_labels | component_images | defect_originals
    
    matches = {
        'perfect_matches': set(),  # In all three datasets
        'missing_from_defect': set(),  # In component data but not defect
        'missing_from_component': set(),  # In defect but not component
        'orphaned_labels': set(),  # Labels without images
        'orphaned_images': set(),  # Images without labels
    }
    
    for name in all_names:
        has_comp_label = name in component_labels
        has_comp_image = name in component_images  
        has_defect = name in defect_originals
        
        if has_comp_label and has_comp_image and has_defect:
            matches['perfect_matches'].add(name)
        elif (has_comp_label or has_comp_image) and not has_defect:
            matches['missing_from_defect'].add(name)
        elif has_defect and not (has_comp_label or has_comp_image):
            matches['missing_from_component'].add(name)
        elif has_comp_label and not has_comp_image:
            matches['orphaned_labels'].add(name)
        elif has_comp_image and not has_comp_label:
            matches['orphaned_images'].add(name)
    
    print(f"   ✅ Perfect matches: {len(matches['perfect_matches'])}")
    print(f"   ❌ Missing from defect: {len(matches['missing_from_defect'])}")
    print(f"   ❌ Missing from component: {len(matches['missing_from_component'])}")
    print(f"   ⚠️  Orphaned labels: {len(matches['orphaned_labels'])}")
    print(f"   ⚠️  Orphaned images: {len(matches['orphaned_images'])}")
    
    return matches

def generate_mismatch_reports(base_path, matching_results, mismatched_dir):
    """Generate text reports and move mismatched files"""
    
    # Generate text reports
    reports = {
        'missing_from_defect.txt': matching_results['missing_from_defect'],
        'missing_from_component.txt': matching_results['missing_from_component'],
        'orphaned_labels.txt': matching_results['orphaned_labels'],
        'orphaned_images.txt': matching_results['orphaned_images']
    }
    
    for filename, items in reports.items():
        report_path = base_path / filename
        with open(report_path, 'w') as f:
            f.write(f"# {filename.replace('.txt', '').replace('_', ' ').title()}\n")
            f.write(f"# Generated during Phase 1 validation\n")
            f.write(f"# Total items: {len(items)}\n\n")
            
            for item in sorted(items):
                f.write(f"{item}\n")
        
        print(f"   📝 Generated {filename} with {len(items)} items")
    
    # Move mismatched files to separate directory
    component_labels_dir = base_path / "component_labels"
    component_images_dir = base_path / "component_images"
    
    moved_count = 0
    
    # Move orphaned label files
    for name in matching_results['orphaned_labels']:
        src = component_labels_dir / f"{name}.json"
        if src.exists():
            dst = mismatched_dir / f"{name}.json"
            shutil.move(str(src), str(dst))
            moved_count += 1
    
    # Move orphaned image files  
    for name in matching_results['orphaned_images']:
        src = component_images_dir / f"{name}.jpg"
        if src.exists():
            dst = mismatched_dir / f"{name}.jpg"
            shutil.move(str(src), str(dst))
            moved_count += 1
    
    print(f"   📁 Moved {moved_count} mismatched files to {mismatched_dir}")

def create_matched_dataset_list(base_path, matching_results):
    """Create final list of perfectly matched images"""
    matched_images = sorted(matching_results['perfect_matches'])
    
    matched_file = base_path / "matched_images.txt"
    with open(matched_file, 'w') as f:
        f.write("# Final Matched Dataset\n")
        f.write("# Images that exist in component_labels, component_images, and defect_data\n")
        f.write(f"# Total: {len(matched_images)} images\n\n")
        
        for img_name in matched_images:
            f.write(f"{img_name}\n")
    
    print(f"   ✅ Saved {len(matched_images)} matched images to matched_images.txt")
    
    return matched_images

def generate_exploration_summary(base_path, component_exploration, defect_exploration, matching_results):
    """Generate comprehensive exploration summary"""
    
    summary_file = base_path / "exploration_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("# DATASET EXPLORATION SUMMARY\n")
        f.write("# Generated during Phase 1 validation\n\n")
        
        # Component dataset summary
        f.write("## COMPONENT DATASET\n")
        f.write(f"Total label files: {component_exploration['total_label_files']}\n")
        f.write(f"Total image files: {component_exploration['total_image_files']}\n")
        f.write(f"Component classes found: {dict(component_exploration['component_classes'])}\n")
        
        if component_exploration['confidence_stats']:
            f.write(f"Confidence range: {component_exploration['confidence_min']:.3f} - {component_exploration['confidence_max']:.3f}\n")
            f.write(f"Average confidence: {component_exploration['confidence_avg']:.3f}\n")
        
        # Image dimensions analysis
        if component_exploration['image_dimensions']:
            dims = component_exploration['image_dimensions']
            unique_dims = list(set(dims))
            f.write(f"Image dimensions found: {unique_dims}\n")
        
        f.write(f"Sample annotations:\n")
        for i, sample in enumerate(component_exploration['sample_annotations']):
            f.write(f"  Sample {i+1}: {sample['file']}\n")
            f.write(f"    Image dims: {sample['image_dims']}\n") 
            f.write(f"    Predictions: {sample['num_predictions']}\n")
        
        # Defect dataset summary  
        f.write(f"\n## DEFECT DATASET\n")
        f.write(f"Total annotations: {defect_exploration['total_annotations']}\n")
        f.write(f"Total images in COCO: {defect_exploration['total_images_in_coco']}\n")
        f.write(f"Total image files: {defect_exploration['total_image_files']}\n")
        f.write(f"Defect categories: {defect_exploration['defect_categories']}\n")
        f.write(f"Defect counts: {dict(defect_exploration['defect_counts'])}\n")
        
        # Matching summary
        f.write(f"\n## DATASET MATCHING\n")
        for key, items in matching_results.items():
            f.write(f"{key}: {len(items)} items\n")
        
        # Errors
        all_errors = component_exploration['errors'] + defect_exploration['errors']
        if all_errors:
            f.write(f"\n## ERRORS ENCOUNTERED\n")
            for error in all_errors:
                f.write(f"- {error}\n")
    
    print(f"   📊 Generated comprehensive summary in exploration_summary.txt")

# Main execution
if __name__ == "__main__":
    # Set your base path here
    BASE_PATH = "/Users/sandhyanayar/Development/PCBAHackathon/defect_model_data"
    
    matched_images, component_data, defect_data = explore_and_validate_datasets(BASE_PATH)