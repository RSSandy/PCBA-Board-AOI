import os
from PIL import Image
import pillow_heif
import sys

def convert_heic_to_jpg(folder_path):
    pillow_heif.register_heif_opener()

    output_folder = os.path.join(folder_path, "converted_jpgs")
    os.makedirs(output_folder, exist_ok=True)

    total_found = 0
    total_converted = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(".heic"):
                total_found += 1
                heic_path = file_path
                jpg_name = os.path.splitext(file)[0] + ".jpg"
                jpg_path = os.path.join(output_folder, jpg_name)

                print(f"🖼 Found HEIC: {heic_path}")

                try:
                    image = Image.open(heic_path).convert("RGB")
                    image.save(jpg_path, "JPEG")
                    print(f"✅ Converted → {jpg_path}")
                    total_converted += 1
                except Exception as e:
                    print(f"❌ Failed to convert {heic_path}: {e}")

    print(f"\nSummary: Found {total_found} HEIC files, converted {total_converted} successfully.")
    if total_found == 0:
        print("⚠️ No .heic files detected — check your folder or extension spelling.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python to_jpg.py <folder_path>")
        sys.exit(1)

    folder = os.path.abspath(sys.argv[1])
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory.")
        sys.exit(1)

    convert_heic_to_jpg(folder)
