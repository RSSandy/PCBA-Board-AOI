"""
run_pipeline.py - PCBA Defect Detection Pipeline Orchestrator

PURPOSE:
Orchestrates the complete PCBA defect detection pipeline from image capture to final report.

PIPELINE STAGES:
1. Image capture (main.py) [optional]
2. Image preprocessing (preprocessing.py)
3. Component detection (yolo_detection.py)
4. Grid generation and patch extraction (grid_processor.py)
5. Defect inference (defect_inference.py)
6. Report generation (report_generator.py)

DEPENDENCIES:
- All pipeline modules
- YOLO weights at ./models/weights.pt
- TFLite model at ./models/pcba_defect_model.tflite

USAGE:
python3 run_pipeline.py --image test.jpg
python3 run_pipeline.py --capture --image board_001.jpg
python3 run_pipeline.py --image test.jpg --output-dir results/
"""

import subprocess
import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime

class PCBAPipeline:
    def __init__(self, image_path, output_dir="results", skip_capture=True):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.skip_capture = skip_capture
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate unique run ID
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"Pipeline run ID: {self.run_id}")
        print(f"Output directory: {self.run_dir}")

    def log_stage(self, stage_name, status="RUNNING", details=None):
        """Log pipeline stage status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {stage_name}: {status}")
        if details:
            print(f"    {details}")

    def run_command(self, command, stage_name, required_output=None):
        """Run a pipeline command and handle errors"""
        self.log_stage(stage_name, "RUNNING", f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True,
                cwd="."
            )
            
            if required_output and not Path(required_output).exists():
                raise FileNotFoundError(f"Expected output not found: {required_output}")
            
            self.log_stage(stage_name, "SUCCESS")
            return result
            
        except subprocess.CalledProcessError as e:
            self.log_stage(stage_name, "FAILED", f"Exit code: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
        except Exception as e:
            self.log_stage(stage_name, "FAILED", str(e))
            raise

    def stage_1_capture(self):
        """Stage 1: Capture image (optional)"""
        if self.skip_capture:
            if not self.image_path.exists():
                raise FileNotFoundError(f"Input image not found: {self.image_path}")
            self.log_stage("STAGE 1: Image Capture", "SKIPPED", f"Using existing: {self.image_path}")
            return str(self.image_path)
        
        self.run_command(
            ["python3", "main.py", "--output", str(self.image_path)],
            "STAGE 1: Image Capture",
            str(self.image_path)
        )
        return str(self.image_path)

    def stage_2_preprocessing(self, input_image):
        """Stage 2: Image preprocessing"""
        output_image = self.run_dir / f"{self.image_path.stem}_preprocessed.jpg"
        
        self.run_command(
            ["python3", "preprocessing.py", "--input", input_image, "--output", str(output_image)],
            "STAGE 2: Preprocessing",
            str(output_image)
        )
        return str(output_image)

    def stage_3_yolo_detection(self, preprocessed_image):
        """Stage 3: YOLO component detection"""
        output_json = self.run_dir / f"{self.image_path.stem}_components.json"
        
        self.run_command(
            ["python3", "yolo_detection.py", 
             "--image", preprocessed_image,
             "--weights", "./models/weights.pt",
             "--output", str(output_json)],
            "STAGE 3: Component Detection",
            str(output_json)
        )
        return str(output_json)

    def stage_4_grid_processing(self, preprocessed_image, components_json):
        """Stage 4: Grid generation and patch extraction"""
        patches_dir = self.run_dir / "patches"
        patches_dir.mkdir(exist_ok=True)
        
        output_file = patches_dir / "patches_metadata.json"
        
        self.run_command(
            ["python3", "grid_processor.py",
             "--image", preprocessed_image,
             "--components", components_json,
             "--output-dir", str(patches_dir)],
            "STAGE 4: Grid Processing",
            str(output_file)
        )
        return str(patches_dir)

    def stage_5_defect_inference(self, patches_dir):
        """Stage 5: Defect detection inference"""
        results_file = self.run_dir / "defect_predictions.json"
        
        self.run_command(
            ["python3", "defect_inference.py",
             "--patches-dir", patches_dir,
             "--model", "./models/pcba_defect_model.tflite",
             "--output", str(results_file)],
            "STAGE 5: Defect Inference",
            str(results_file)
        )
        return str(results_file)

    def stage_6_report_generation(self, defect_results):
        """Stage 6: Generate final report"""
        report_file = self.run_dir / "defect_report.json"
        html_report = self.run_dir / "defect_report.html"
        
        self.run_command(
            ["python3", "report_generator.py",
             "--results", defect_results,
             "--run-dir", str(self.run_dir),
             "--output", str(report_file),
             "--html", str(html_report)],
            "STAGE 6: Report Generation",
            str(report_file)
        )
        return str(report_file), str(html_report)

    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("=" * 60)
        print("PCBA DEFECT DETECTION PIPELINE")
        print("=" * 60)
        
        pipeline_start = time.time()
        
        try:
            # Stage 1: Image Capture
            input_image = self.stage_1_capture()
            
            # Stage 2: Preprocessing  
            preprocessed_image = self.stage_2_preprocessing(input_image)
            
            # Stage 3: YOLO Component Detection
            components_json = self.stage_3_yolo_detection(preprocessed_image)
            
            # Stage 4: Grid Processing
            patches_dir = self.stage_4_grid_processing(preprocessed_image, components_json)
            
            # Stage 5: Defect Inference
            defect_results = self.stage_5_defect_inference(patches_dir)
            
            # Stage 6: Report Generation
            report_file, html_report = self.stage_6_report_generation(defect_results)
            
            # Pipeline completion
            pipeline_time = time.time() - pipeline_start
            
            print("=" * 60)
            self.log_stage("PIPELINE COMPLETE", "SUCCESS", f"Total time: {pipeline_time:.2f}s")
            print(f"Results saved to: {self.run_dir}")
            print(f"Final report: {report_file}")
            print(f"HTML report: {html_report}")
            print("=" * 60)
            
            return {
                "status": "success",
                "run_id": self.run_id,
                "output_dir": str(self.run_dir),
                "reports": {
                    "json": str(report_file),
                    "html": str(html_report)
                },
                "pipeline_time": pipeline_time
            }
            
        except Exception as e:
            pipeline_time = time.time() - pipeline_start
            print("=" * 60)
            self.log_stage("PIPELINE FAILED", "ERROR", str(e))
            print(f"Partial results may be in: {self.run_dir}")
            print("=" * 60)
            
            return {
                "status": "failed",
                "error": str(e),
                "run_id": self.run_id,
                "output_dir": str(self.run_dir),
                "pipeline_time": pipeline_time
            }

def check_dependencies():
    """Check if all required files and dependencies exist"""
    required_files = [
        "main.py",
        "preprocessing.py", 
        "yolo_detection.py",
        "grid_processor.py",
        "defect_inference.py",
        "report_generator.py",
        "./models/weights.pt",
        "./models/pcba_defect_model.tflite"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="PCBA Defect Detection Pipeline")
    parser.add_argument("--image", required=True, help="Input image file path")
    parser.add_argument("--capture", action="store_true", help="Capture new image (overrides existing)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps or not check_dependencies():
        if not check_dependencies():
            print("Dependency check failed!")
            return 1
        else:
            print("All dependencies found!")
            return 0
    
    # Run pipeline
    pipeline = PCBAPipeline(
        image_path=args.image,
        output_dir=args.output_dir,
        skip_capture=not args.capture
    )
    
    result = pipeline.run_complete_pipeline()
    
    return 0 if result["status"] == "success" else 1

if __name__ == "__main__":
    sys.exit(main())