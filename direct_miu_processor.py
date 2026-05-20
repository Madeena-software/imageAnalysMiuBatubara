#!/usr/bin/env python3
"""
Direct TIFF image processor - extracts MIU values and generates reports.
No Playwright needed - processes images directly using the existing Python modules.
"""

import sys
import os
import csv
from pathlib import Path
from datetime import datetime

# Add the app modules to path
APP_DIR = Path(__file__).parent / "public" / "image-analysis-miu-batubara"
sys.path.insert(0, str(APP_DIR))

try:
    from circle_detection import process_tiff_image, detect_grid_from_diagonal, analyze_grid_histograms, compare_diagonals
    from block_detection import process_blocks, analyze_block_histograms
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

class DirectMIUProcessor:
    """Process TIFF images directly without browser automation."""
    
    def __init__(self):
        # Handle both Windows and WSL paths
        if os.path.exists("/mnt/c/Users"):
            self.image_folder = "/mnt/c/Users/ASUS/Desktop/BatuBara Paiton/grabber-processed-tiff"
            self.downloads_folder = "/mnt/c/Users/ASUS/Downloads/MIU_Analysis_Results"
        else:
            self.image_folder = r"C:\Users\ASUS\Desktop\BatuBara Paiton\grabber-processed-tiff"
            self.downloads_folder = os.path.expanduser(r"~\Downloads\MIU_Analysis_Results")
        
        Path(self.downloads_folder).mkdir(parents=True, exist_ok=True)
        self.results = []
        
        print(f"📁 Image folder: {self.image_folder}")
        print(f"📊 Output folder: {self.downloads_folder}")
    
    def get_tiff_files(self):
        """Get all TIFF files from the folder."""
        if not os.path.isdir(self.image_folder):
            print(f"❌ Folder not found: {self.image_folder}")
            return []
        
        files = sorted([
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.tiff', '.tif'))
        ])
        
        print(f"📁 Found {len(files)} TIFF files:")
        for f in files:
            print(f"   • {os.path.basename(f)}")
        
        return files
    
    def process_image(self, file_path):
        """Process a single TIFF image."""
        filename = os.path.basename(file_path)
        print(f"\n⚙️  Processing: {filename}")
        
        try:
            # Read file
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Default processing params for block mode
            params = {
                "threshold_value": 54000,
                "min_length": 1200,
                "max_length": 1600,
                "min_rectangularity": 0.9,
                "min_solidity": 0.9,
                "data_type": "uint16"
            }
            
            # Process blocks
            print("   Processing blocks...")
            blocks_result = process_blocks(file_bytes, params)
            
            if not blocks_result:
                print(f"   ⚠️  Block processing returned empty result")
                return None
            
            # Extract MIU values from results
            result = {
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "upper_mu": None,
                "lower_mu": None,
                "upper_mu_std": None,
                "lower_mu_std": None,
            }
            
            # Try to extract from blocks_result
            if isinstance(blocks_result, dict):
                # Check for MIU values in different possible locations
                if "upper_mu" in blocks_result:
                    result["upper_mu"] = blocks_result.get("upper_mu")
                if "lower_mu" in blocks_result:
                    result["lower_mu"] = blocks_result.get("lower_mu")
                if "upper_mu_std" in blocks_result:
                    result["upper_mu_std"] = blocks_result.get("upper_mu_std")
                if "lower_mu_std" in blocks_result:
                    result["lower_mu_std"] = blocks_result.get("lower_mu_std")
                
                # Alternative keys
                if "left_mu" in blocks_result:
                    result["upper_mu"] = blocks_result.get("left_mu")
                if "right_mu" in blocks_result:
                    result["lower_mu"] = blocks_result.get("right_mu")
            
            print(f"   ✅ Processed successfully")
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_csv(self):
        """Save results to CSV."""
        csv_path = os.path.join(self.downloads_folder, "MIU_Results.csv")
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "Filename", "Upper MIU", "Upper MIU STD",
                    "Lower MIU", "Lower MIU STD", "Timestamp"
                ])
                writer.writeheader()
                
                for result in self.results:
                    if result:
                        writer.writerow({
                            "Filename": result["filename"],
                            "Upper MIU": result["upper_mu"],
                            "Upper MIU STD": result["upper_mu_std"],
                            "Lower MIU": result["lower_mu"],
                            "Lower MIU STD": result["lower_mu_std"],
                            "Timestamp": result["timestamp"]
                        })
            
            print(f"\n✅ CSV saved: {csv_path}")
        except Exception as e:
            print(f"❌ Failed to save CSV: {e}")
    
    def run(self):
        """Run the processor."""
        print("="*60)
        print("🎯 Direct MIU Processor (No Browser Needed)")
        print("="*60)
        
        files = self.get_tiff_files()
        if not files:
            print("❌ No TIFF files found")
            return
        
        total = len(files)
        for idx, file_path in enumerate(files, 1):
            print(f"\n[{idx}/{total}]", end="")
            result = self.process_image(file_path)
            if result:
                self.results.append(result)
        
        self.save_csv()
        
        print("\n" + "="*60)
        print("✅ Processing Complete!")
        print("="*60)
        print(f"📊 Results: {len(self.results)}/{total} images processed")
        print(f"💾 CSV location: {os.path.join(self.downloads_folder, 'MIU_Results.csv')}")


if __name__ == "__main__":
    processor = DirectMIUProcessor()
    processor.run()
