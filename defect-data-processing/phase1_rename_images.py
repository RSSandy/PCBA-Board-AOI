"""
Quick script to rename component image files by removing the '_output_image_v2' suffix
This fixes the file not found errors in Phase 2
"""

import os
from pathlib import Path

def rename_component_images(component_images_dir):
    """Rename files from IMG_XXXX.jpg_output_image_v2.jpg to IMG_XXXX.jpg"""
    
    component_images_dir = Path(component_images_dir)
    
    if not component_images_dir.exists():
        print(f"Directory not found: {component_images_dir}")
        return
    
    renamed_count = 0
    
    # Find all files with the problematic suffix
    for file_path in component_images_dir.glob("*.jpg_output_image_v2.jpg"):
        # Create new filename by removing '_output_image_v2'
        old_name = file_path.name
        new_name = old_name.replace('.jpg_output_image_v2.jpg', '.jpg')
        new_path = component_images_dir / new_name
        
        # Check if target already exists
        if new_path.exists():
            print(f"⚠️  Target already exists, skipping: {new_name}")
            continue
        
        # Rename the file
        file_path.rename(new_path)
        print(f"✅ Renamed: {old_name} → {new_name}")
        renamed_count += 1
    
    print(f"\n🎉 Renamed {renamed_count} files")

if __name__ == "__main__":
    BASE_PATH = "/Users/sandhyanayar/Development/PCBAHackathon/defect_model_data"
    COMPONENT_IMAGES_DIR = f"{BASE_PATH}/component_images"
    
    print("=== RENAMING COMPONENT IMAGE FILES ===")
    rename_component_images(COMPONENT_IMAGES_DIR)