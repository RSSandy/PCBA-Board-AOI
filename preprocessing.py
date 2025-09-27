"""
preprocessing.py - PCBA Image Preprocessing with CLI support

PURPOSE:
Preprocesses PCBA images using the crop_pcba_to_jpg function for optimal component detection.
Handles PCBA detection, perspective correction, and image enhancement.

INPUTS:
- Raw PCBA image (JPG/PNG)

OUTPUTS:
- Preprocessed image ready for YOLO detection

DEPENDENCIES:
- opencv-python
- numpy

USAGE:
python3 preprocessing.py --input test.jpg --output test_preprocessed.jpg
"""

import cv2
import numpy as np
import os
import argparse
import sys
from pathlib import Path

def crop_pcba_to_jpg(
    image_path: str,
    output_path: str,
    *,
    prefer: str = "auto",
    min_area_frac: float = 0.12,
    max_aspect_ratio: float = 4.0,
) -> str:
    """
    Detect and crop PCBA from image with perspective correction
    
    Args:
        image_path: Input image path
        output_path: Output image path
        prefer: Detection method ('auto', 'color', 'geometry')
        min_area_frac: Minimum area fraction of image for valid PCBA
        max_aspect_ratio: Maximum aspect ratio for valid PCBA
    
    Returns:
        str: Path to output image
    """

    def resize_for_processing(img, max_dim=1600):
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img, 1.0
        s = max_dim / max(h, w)
        return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA), s

    def order_quad(pts):
        pts = pts.astype(np.float32).reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        rect[0] = pts[np.argmin(s)]  # tl
        rect[2] = pts[np.argmax(s)]  # br
        rect[1] = pts[np.argmin(d)]  # tr
        rect[3] = pts[np.argmax(d)]  # bl
        return rect

    def warp(img, quad):
        rect = order_quad(quad)
        tl, tr, br, bl = rect
        wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
        W = max(int(max(wA, wB)), 1); H = max(int(max(hA, hB)), 1)
        M = cv2.getPerspectiveTransform(rect, np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32))
        return cv2.warpPerspective(img, M, (W, H))

    def approx_quad(cnt):
        peri = cv2.arcLength(cnt, True)
        return cv2.approxPolyDP(cnt, 0.02 * peri, True)

    def min_area_rect_quad(cnt):
        return cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)

    def detect_by_color(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ranges = [((35,40,30),(90,255,255)), ((90,40,30),(140,255,255)), ((0,50,40),(10,255,255)), ((160,50,40),(179,255,255))]
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, 1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        best = max(cnts, key=cv2.contourArea)
        ap = approx_quad(best)
        return ap.reshape(-1,2).astype(np.float32) if len(ap) == 4 else min_area_rect_quad(best)

    def detect_by_geometry(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), 1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        best = max(cnts, key=cv2.contourArea)
        ap = approx_quad(best)
        return ap.reshape(-1,2).astype(np.float32) if len(ap) == 4 else min_area_rect_quad(best)

    def passes_safeguards(q, img):
        if q is None:
            return False

        # normalize quad
        q = np.asarray(q, dtype=np.float32).reshape(-1, 2)
        if q.shape != (4, 2) or not np.isfinite(q).all():
            return False

        # get height/width from the IMAGE, not by slicing the array
        h, w = img.shape[:2]

        # area check (wrap to contour shape and cast to float)
        contour = q.reshape((-1, 1, 2)).astype(np.float32)
        area = float(cv2.contourArea(contour))
        if area < float(min_area_frac) * float(h) * float(w):
            return False

        # aspect-ratio check
        rect = order_quad(q)
        tl, tr, br, bl = rect
        wA = np.linalg.norm(br - bl);
        wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br);
        hB = np.linalg.norm(tl - bl)
        W = float(max(wA, wB));
        H = float(max(hA, hB))
        ar = (W / (H + 1e-6)) if W >= H else (H / (W + 1e-6))

        return ar <= float(max_aspect_ratio)

    # ---- pipeline ----
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    proc, scale = resize_for_processing(orig)

    quad = None
    if prefer in ("auto", "color"):
        quad = detect_by_color(proc)
    if quad is None and prefer in ("auto", "geometry"):
        quad = detect_by_geometry(proc)

    if quad is not None and passes_safeguards(quad, proc):
        quad = (np.asarray(quad, dtype=np.float32) / float(scale)).astype(np.float32)
        cropped = warp(orig, quad)
        print(f"PCBA detected and perspective corrected")
    else:
        cropped = orig
        print(f"No valid PCBA detected, using original image")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return output_path

def enhance_image(image_path: str, output_path: str) -> str:
    """
    Additional image enhancement for better component detection
    
    Args:
        image_path: Input image path
        output_path: Output enhanced image path
    
    Returns:
        str: Path to enhanced image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to LAB for better lighting control
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Light denoising while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Save enhanced image
    cv2.imwrite(output_path, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    print(f"Image enhanced and saved to: {output_path}")
    return output_path

def main():
    """Main preprocessing function with CLI support"""
    parser = argparse.ArgumentParser(description="Preprocess PCBA images for component detection")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--method", choices=["auto", "color", "geometry"], default="auto",
                       help="PCBA detection method")
    parser.add_argument("--enhance", action="store_true", 
                       help="Apply additional image enhancement")
    parser.add_argument("--min-area", type=float, default=0.12,
                       help="Minimum area fraction for valid PCBA detection")
    parser.add_argument("--max-aspect", type=float, default=4.0,
                       help="Maximum aspect ratio for valid PCBA")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input image not found: {args.input}")
        return 1
    
    try:
        print(f"Processing image: {args.input}")
        print(f"Detection method: {args.method}")
        
        # Step 1: PCBA detection and perspective correction
        if args.enhance:
            # Use temporary file for intermediate processing
            temp_path = str(Path(args.output).parent / f"temp_{Path(args.output).name}")
            crop_pcba_to_jpg(
                args.input, 
                temp_path,
                prefer=args.method,
                min_area_frac=args.min_area,
                max_aspect_ratio=args.max_aspect
            )
            
            # Step 2: Additional enhancement
            enhance_image(temp_path, args.output)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
        else:
            # Just PCBA detection and cropping
            crop_pcba_to_jpg(
                args.input, 
                args.output,
                prefer=args.method,
                min_area_frac=args.min_area,
                max_aspect_ratio=args.max_aspect
            )
        
        # Verify output
        output_img = cv2.imread(args.output)
        if output_img is None:
            raise ValueError("Failed to create output image")
        
        height, width = output_img.shape[:2]
        print(f"Preprocessing complete!")
        print(f"Output: {args.output}")
        print(f"Output size: {width}x{height} pixels")
        
        return 0
        
    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())