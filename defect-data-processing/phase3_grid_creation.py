"""
PHASE 3: GRID GENERATION & MULTI-LABEL ASSIGNMENT

This script takes the consolidated annotations from Phase 2 and generates a systematic grid of 
224x224 patches with multi-label assignments for training.

Key Functions:
1. Generate systematic 224x224 patches with 50% overlap (112-pixel stride)
2. Create multi-label system for each patch:
   - Component labels: has_led, has_resistor, has_capacitor, etc. (binary for each component type)
   - Defect labels: has_dirt, has_missing, has_rotate, has_solder (binary for each defect type)  
   - Meta labels: has_any_component, has_any_defect, is_background
3. Calculate overlap between patches and bounding boxes
4. Assign labels based on overlap thresholds (default 30%)
5. Track overlap statistics and flag complex multi-label cases
6. Handle edge cases at image boundaries

Input:
- consolidated_data.json (from Phase 2)
- Original image files in /component_images/

Output:
- patch_dataset.npz (numpy arrays of patches and labels)
- patch_metadata.csv (detailed info about each patch)
- overlap_analysis.json (statistics about overlap thresholds)
- multilabel_complexity_report.txt (patches with many labels)

Label Structure per Patch:
[has_led, has_resistor, has_capacitor, has_ic, has_connector, has_diode, has_transistor, 
 has_inductor, has_relay, has_potentiometer, has_dirt, has_missing, has_rotate, has_solder, 
 has_any_component, has_any_defect, is_background]

Example: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0] = LED with missing defect
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def generate_patches_and_labels(base_path, patch_size=224, stride=112, overlap_threshold=0.3):
    """Main function to generate patches and assign multi-labels"""
    
    base_path = Path(base_path)
    
    print("=== PHASE 3: GRID GENERATION & MULTI-LABEL ASSIGNMENT ===\n")
    
    # Load consolidated data from Phase 2
    print("1. Loading consolidated data from Phase 2...")
    consolidated_data = load_consolidated_data(base_path)
    
    # Define label structure
    print("2. Defining multi-label structure...")
    label_structure = define_label_structure(consolidated_data)
    
    # Generate patches for all images
    print(f"3. Generating {patch_size}x{patch_size} patches with {stride}-pixel stride...")
    all_patches, all_labels, all_metadata = generate_all_patches(
        consolidated_data, patch_size, stride, overlap_threshold, label_structure
    )
    
    # Analyze overlap effectiveness
    print("4. Analyzing overlap threshold effectiveness...")
    overlap_analysis = analyze_overlap_effectiveness(all_metadata, overlap_threshold)
    
    # Flag complex multi-label cases
    print("5. Analyzing multi-label complexity...")
    complexity_analysis = analyze_multilabel_complexity(all_labels, all_metadata, label_structure)
    
    # Save all outputs
    print("6. Saving patch dataset and analysis...")
    save_patch_dataset(base_path, all_patches, all_labels, all_metadata, label_structure, 
                      overlap_analysis, complexity_analysis)
    
    print(f"\n✅ Phase 3 Complete!")
    print(f"📊 Generated {len(all_patches)} patches from {len(consolidated_data)} images")
    print(f"📁 Check {base_path / 'patch_dataset.npz'} for training data")
    print(f"📁 Check {base_path / 'patch_metadata.csv'} for detailed patch info")
    
    return all_patches, all_labels, all_metadata, label_structure

def load_consolidated_data(base_path):
    """Load consolidated data from Phase 2"""
    consolidated_file = base_path / "consolidated_data.json"
    
    if not consolidated_file.exists():
        raise FileNotFoundError(f"consolidated_data.json not found. Run Phase 2 first.")
    
    with open(consolidated_file, 'r') as f:
        consolidated_data = json.load(f)
    
    print(f"   Loaded data for {len(consolidated_data)} images")
    return consolidated_data

def define_label_structure(consolidated_data):
    """Define the multi-label structure based on actual data"""
    
    # Extract all unique component classes
    component_classes = set()
    defect_categories = set()
    
    for img_data in consolidated_data.values():
        for comp in img_data['components']:
            component_classes.add(comp['class'].lower())
        for defect in img_data['defects']:
            defect_categories.add(defect['category'])
    
    component_classes = sorted(list(component_classes))
    defect_categories = sorted(list(defect_categories))
    
    # Create label structure
    component_labels = [f"has_{cls}" for cls in component_classes]
    defect_labels = [f"has_{cat}" for cat in defect_categories if cat != 'defects']  # Skip generic 'defects'
    meta_labels = ['has_any_component', 'has_any_defect', 'is_background']
    
    all_labels = component_labels + defect_labels + meta_labels
    
    label_structure = {
        'component_classes': component_classes,
        'defect_categories': [cat for cat in defect_categories if cat != 'defects'],
        'component_labels': component_labels,
        'defect_labels': defect_labels,
        'meta_labels': meta_labels,
        'all_labels': all_labels,
        'total_labels': len(all_labels)
    }
    
    print(f"   Component classes: {component_classes}")
    print(f"   Defect categories: {label_structure['defect_categories']}")
    print(f"   Total label dimensions: {len(all_labels)}")
    
    return label_structure

def generate_all_patches(consolidated_data, patch_size, stride, overlap_threshold, label_structure):
    """Generate patches and labels for all images"""
    
    all_patches = []
    all_labels = []
    all_metadata = []
    
    for img_name, img_data in consolidated_data.items():
        print(f"   Processing {img_name}...")
        
        # Load image
        img = cv2.imread(img_data['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate patches for this image
        patches, labels, metadata = generate_patches_for_image(
            img, img_name, img_data, patch_size, stride, overlap_threshold, label_structure
        )
        
        all_patches.extend(patches)
        all_labels.extend(labels)
        all_metadata.extend(metadata)
        
        print(f"     Generated {len(patches)} patches")
    
    return np.array(all_patches), np.array(all_labels), all_metadata

def generate_patches_for_image(img, img_name, img_data, patch_size, stride, overlap_threshold, label_structure):
    """Generate patches and labels for a single image"""
    
    height, width = img.shape[:2]
    patches = []
    labels = []
    metadata = []
    
    # Calculate number of patches
    num_patches_x = (width - patch_size) // stride + 1
    num_patches_y = (height - patch_size) // stride + 1
    
    for y_idx in range(num_patches_y):
        for x_idx in range(num_patches_x):
            # Calculate patch coordinates
            x_start = x_idx * stride
            y_start = y_idx * stride
            x_end = min(x_start + patch_size, width)
            y_end = min(y_start + patch_size, height)
            
            # Handle edge cases - skip partial patches for now
            if x_end - x_start < patch_size or y_end - y_start < patch_size:
                continue
            
            # Extract patch
            patch = img[y_start:y_end, x_start:x_end]
            patches.append(patch)
            
            # Calculate labels for this patch
            patch_bbox = [x_start, y_start, patch_size, patch_size]
            patch_labels, patch_metadata = calculate_patch_labels(
                patch_bbox, img_data, overlap_threshold, label_structure
            )
            
            labels.append(patch_labels)
            
            # Store metadata
            metadata.append({
                'image_name': img_name,
                'patch_coordinates': [x_start, y_start, x_end, y_end],
                'patch_index': [x_idx, y_idx],
                'overlapping_components': patch_metadata['overlapping_components'],
                'overlapping_defects': patch_metadata['overlapping_defects'],
                'component_overlaps': patch_metadata['component_overlaps'],
                'defect_overlaps': patch_metadata['defect_overlaps'],
                'total_overlaps': len(patch_metadata['overlapping_components']) + len(patch_metadata['overlapping_defects'])
            })
    
    return patches, labels, metadata

def calculate_patch_labels(patch_bbox, img_data, overlap_threshold, label_structure):
    """Calculate multi-label vector for a single patch"""
    
    # Initialize label vector
    label_vector = np.zeros(label_structure['total_labels'], dtype=int)
    
    # Track overlaps for metadata
    overlapping_components = []
    overlapping_defects = []
    component_overlaps = []
    defect_overlaps = []
    
    # Check component overlaps
    for comp in img_data['components']:
        overlap_pct = calculate_bbox_overlap_percentage(patch_bbox, comp['bbox'])
        
        if overlap_pct > overlap_threshold:
            overlapping_components.append(comp)
            component_overlaps.append(overlap_pct)
            
            # Set component label
            comp_class = comp['class'].lower()
            if comp_class in label_structure['component_classes']:
                comp_idx = label_structure['component_labels'].index(f"has_{comp_class}")
                label_vector[comp_idx] = 1
    
    # Check defect overlaps
    for defect in img_data['defects']:
        overlap_pct = calculate_bbox_overlap_percentage(patch_bbox, defect['bbox'])
        
        if overlap_pct > overlap_threshold:
            overlapping_defects.append(defect)
            defect_overlaps.append(overlap_pct)
            
            # Set defect label
            defect_cat = defect['category']
            if defect_cat in label_structure['defect_categories']:
                defect_idx = len(label_structure['component_labels']) + label_structure['defect_categories'].index(defect_cat)
                label_vector[defect_idx] = 1
    
    # Set meta labels
    meta_start_idx = len(label_structure['component_labels']) + len(label_structure['defect_labels'])
    
    # has_any_component
    if len(overlapping_components) > 0:
        label_vector[meta_start_idx] = 1
    
    # has_any_defect  
    if len(overlapping_defects) > 0:
        label_vector[meta_start_idx + 1] = 1
    
    # is_background
    if len(overlapping_components) == 0 and len(overlapping_defects) == 0:
        label_vector[meta_start_idx + 2] = 1
    
    metadata = {
        'overlapping_components': overlapping_components,
        'overlapping_defects': overlapping_defects,
        'component_overlaps': component_overlaps,
        'defect_overlaps': defect_overlaps
    }
    
    return label_vector, metadata

def calculate_bbox_overlap_percentage(patch_bbox, annotation_bbox):
    """Calculate what percentage of the patch overlaps with an annotation"""
    
    px1, py1, pw, ph = patch_bbox
    px2, py2 = px1 + pw, py1 + ph
    
    ax1, ay1, aw, ah = annotation_bbox
    ax2, ay2 = ax1 + aw, ay1 + ah
    
    # Calculate intersection
    left = max(px1, ax1)
    top = max(py1, ay1)
    right = min(px2, ax2)
    bottom = min(py2, ay2)
    
    if left < right and top < bottom:
        intersection_area = (right - left) * (bottom - top)
        patch_area = pw * ph
        return intersection_area / patch_area if patch_area > 0 else 0
    
    return 0.0

def analyze_overlap_effectiveness(all_metadata, overlap_threshold):
    """Analyze how effective the overlap threshold is"""
    
    all_overlaps = []
    
    for patch_meta in all_metadata:
        all_overlaps.extend(patch_meta['component_overlaps'])
        all_overlaps.extend(patch_meta['defect_overlaps'])
    
    if not all_overlaps:
        return {'error': 'No overlaps found'}
    
    all_overlaps = np.array(all_overlaps)
    
    analysis = {
        'total_overlaps': len(all_overlaps),
        'overlap_threshold': overlap_threshold,
        'overlaps_above_threshold': np.sum(all_overlaps > overlap_threshold),
        'overlaps_below_threshold': np.sum(all_overlaps <= overlap_threshold),
        'overlap_statistics': {
            'min': float(np.min(all_overlaps)),
            'max': float(np.max(all_overlaps)),
            'mean': float(np.mean(all_overlaps)),
            'median': float(np.median(all_overlaps)),
            'std': float(np.std(all_overlaps))
        },
        'threshold_effectiveness': {
            'precision': float(np.sum(all_overlaps > overlap_threshold) / len(all_overlaps)),
            'low_overlaps_caught': float(np.sum((all_overlaps > 0.1) & (all_overlaps <= overlap_threshold)) / len(all_overlaps))
        }
    }
    
    print(f"   Overlap analysis: {analysis['overlaps_above_threshold']}/{analysis['total_overlaps']} above threshold")
    print(f"   Mean overlap: {analysis['overlap_statistics']['mean']:.3f}")
    
    return analysis

def analyze_multilabel_complexity(all_labels, all_metadata, label_structure):
    """Analyze multi-label complexity and flag interesting cases"""
    
    label_counts = np.sum(all_labels, axis=0)
    labels_per_patch = np.sum(all_labels, axis=1)
    
    # Find patches with high label complexity
    high_complexity_patches = []
    for i, (label_count, patch_meta) in enumerate(zip(labels_per_patch, all_metadata)):
        if label_count >= 3:  # 3+ labels is complex
            high_complexity_patches.append({
                'patch_index': i,
                'label_count': int(label_count),
                'image_name': patch_meta['image_name'],
                'patch_coordinates': patch_meta['patch_coordinates'],
                'active_labels': [label_structure['all_labels'][j] for j in range(len(label_structure['all_labels'])) if all_labels[i][j] == 1]
            })
    
    # Background vs non-background distribution
    background_idx = label_structure['all_labels'].index('is_background')
    background_patches = np.sum(all_labels[:, background_idx])
    non_background_patches = len(all_labels) - background_patches
    
    analysis = {
        'total_patches': len(all_labels),
        'label_distribution': {label: int(count) for label, count in zip(label_structure['all_labels'], label_counts)},
        'labels_per_patch_stats': {
            'min': int(np.min(labels_per_patch)),
            'max': int(np.max(labels_per_patch)),
            'mean': float(np.mean(labels_per_patch)),
            'median': float(np.median(labels_per_patch))
        },
        'background_distribution': {
            'background_patches': int(background_patches),
            'non_background_patches': int(non_background_patches),
            'background_percentage': float(background_patches / len(all_labels) * 100)
        },
        'high_complexity_patches': high_complexity_patches[:50]  # Limit to first 50 for readability
    }
    
    print(f"   Background patches: {background_patches}/{len(all_labels)} ({analysis['background_distribution']['background_percentage']:.1f}%)")
    print(f"   High complexity patches (3+ labels): {len(high_complexity_patches)}")
    
    return analysis

def save_patch_dataset(base_path, all_patches, all_labels, all_metadata, label_structure, 
                      overlap_analysis, complexity_analysis):
    """Save all patch dataset outputs"""
    
    # Create output directory
    output_dir = base_path / "processed-training-defect-data"
    output_dir.mkdir(exist_ok=True)
    
    # Save patch dataset as numpy arrays
    dataset_file = output_dir / "patch_dataset.npz"
    np.savez_compressed(
        dataset_file,
        patches=all_patches,
        labels=all_labels,
        label_names=label_structure['all_labels']
    )
    print(f"   📁 Saved patch dataset to {dataset_file}")
    
    # Save metadata as CSV
    metadata_df = pd.DataFrame(all_metadata)
    metadata_file = output_dir / "patch_metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    print(f"   📁 Saved patch metadata to {metadata_file}")
    
    # Save overlap analysis
    overlap_file = output_dir / "overlap_analysis.json"
    with open(overlap_file, 'w') as f:
        json.dump(overlap_analysis, f, indent=2)
    print(f"   📁 Saved overlap analysis to {overlap_file}")
    
    # Save complexity analysis
    complexity_file = output_dir / "multilabel_complexity_report.txt"
    with open(complexity_file, 'w') as f:
        f.write("# MULTI-LABEL COMPLEXITY ANALYSIS\n")
        f.write("# Generated during Phase 3 grid generation\n\n")
        
        f.write(f"## SUMMARY\n")
        f.write(f"Total patches: {complexity_analysis['total_patches']}\n")
        f.write(f"Background patches: {complexity_analysis['background_distribution']['background_patches']} ({complexity_analysis['background_distribution']['background_percentage']:.1f}%)\n")
        f.write(f"High complexity patches: {len(complexity_analysis['high_complexity_patches'])}\n\n")
        
        f.write(f"## LABEL DISTRIBUTION\n")
        for label, count in complexity_analysis['label_distribution'].items():
            percentage = count / complexity_analysis['total_patches'] * 100
            f.write(f"{label}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\n## LABELS PER PATCH STATISTICS\n")
        stats = complexity_analysis['labels_per_patch_stats']
        f.write(f"Min labels per patch: {stats['min']}\n")
        f.write(f"Max labels per patch: {stats['max']}\n") 
        f.write(f"Mean labels per patch: {stats['mean']:.2f}\n")
        f.write(f"Median labels per patch: {stats['median']:.2f}\n")
        
        f.write(f"\n## HIGH COMPLEXITY PATCHES (3+ labels)\n")
        for patch in complexity_analysis['high_complexity_patches']:
            f.write(f"Patch {patch['patch_index']} from {patch['image_name']}: {patch['label_count']} labels\n")
            f.write(f"  Coordinates: {patch['patch_coordinates']}\n")
            f.write(f"  Active labels: {', '.join(patch['active_labels'])}\n\n")
    
    print(f"   📁 Saved complexity analysis to {complexity_file}")
    
    # Save label structure for reference
    label_structure_file = output_dir / "label_structure.json"
    with open(label_structure_file, 'w') as f:
        json.dump(label_structure, f, indent=2)
    print(f"   📁 Saved label structure to {label_structure_file}")

# Main execution
if __name__ == "__main__":
    BASE_PATH = "/Users/sandhyanayar/Development/PCBAHackathon/defect_model_data"
    
    patches, labels, metadata, label_structure = generate_patches_and_labels(BASE_PATH)