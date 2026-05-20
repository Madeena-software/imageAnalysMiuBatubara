#!/usr/bin/env python3
"""
Playwright automation script to batch process TIFF images for MIU analysis.

This script:
1. Starts the local web server
2. Opens the image analysis web app
3. Processes each TIFF file from a Windows folder
4. Extracts MIU and MIU STD values → saves to CSV
5. Downloads PDF analysis reports from the web app
6. Exports everything to Windows Downloads folder

Outputs:
- MIU_Results.csv: Contains Filename, Upper MIU, Upper MIU STD, Lower MIU, Lower MIU STD
- *.pdf files: Individual analysis reports for each image

Usage:
    python automation_miu_processor.py

Requirements:
    - playwright
    - asyncio
"""

import asyncio
import csv
import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import threading
import re

try:
    from playwright.async_api import async_playwright, Browser, Page
except ImportError:
    print("❌ Playwright not installed. Installing...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "playwright"], check=True)
    except:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
    except:
        pass
    from playwright.async_api import async_playwright, Browser, Page


class MIUProcessor:
    """Automates TIFF image processing for MIU analysis."""
    
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self.miu_results: List[Dict] = []
        
        # Handle both Windows and WSL paths
        if os.path.exists("/mnt/c/Users"):
            # Running on WSL - use WSL paths
            self.image_folder = "/mnt/c/Users/ASUS/Desktop/BatuBara Paiton/grabber-processed-tiff"
            self.downloads_folder = "/mnt/c/Users/ASUS/Downloads/MIU_Analysis_Results"
        else:
            # Running on Windows - use Windows paths
            self.image_folder = r"C:\Users\ASUS\Desktop\BatuBara Paiton\grabber-processed-tiff"
            self.downloads_folder = os.path.expanduser(r"~\Downloads\MIU_Analysis_Results")
        
        self.port = 8000
        self.base_url = f"http://localhost:{self.port}"
        self.app_url = f"{self.base_url}/image-analysis-miu-batubara/index.html?mode=block"
        
    def setup_directories(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.downloads_folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ Output folder ready: {self.downloads_folder}")
    
    def start_server(self) -> None:
        """Start the Python HTTP server in a separate thread."""
        print(f"🚀 Starting server on port {self.port}...")
        
        # Find the run.py location
        repo_root = Path(__file__).parent
        run_py = repo_root / "run.py"
        
        if not run_py.exists():
            print(f"❌ run.py not found at {run_py}")
            sys.exit(1)
        
        # Start server process
        self.server_process = subprocess.Popen(
            [sys.executable, str(run_py), str(self.port)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(4)
        
        # Check if process is still running
        if self.server_process.poll() is not None:
            out, err = self.server_process.communicate()
            print(f"❌ Server failed to start")
            if err:
                print(f"Error: {err}")
            if out:
                print(f"Output: {out}")
            sys.exit(1)
        
        print(f"✅ Server started on {self.base_url}")
    
    def stop_server(self) -> None:
        """Stop the HTTP server."""
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("✅ Server stopped")
    
    async def init_browser(self) -> None:
        """Initialize Playwright browser."""
        print("🌐 Initializing browser...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()
        
        # Set download behavior
        await self.page.context.set_extra_http_headers({})
        print("✅ Browser initialized")
    
    async def close_browser(self) -> None:
        """Close the browser."""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            print("✅ Browser closed")
        except Exception as e:
            print(f"⚠️  Error closing browser: {e}")
    
    async def open_app(self) -> None:
        """Open the image analysis web application."""
        print(f"📂 Opening app: {self.app_url}")
        await self.page.goto(self.app_url, wait_until="networkidle")
        
        # Wait for PyScript to initialize
        print("⏳ Waiting for PyScript initialization...")
        try:
            await self.page.wait_for_selector(".upload-section", timeout=30000)
            print("✅ App loaded successfully")
        except Exception as e:
            print(f"⚠️  App may not be fully loaded: {e}")
    
    def get_tiff_files(self) -> List[str]:
        """Get list of TIFF files from the image folder."""
        if not os.path.isdir(self.image_folder):
            print(f"❌ Image folder not found: {self.image_folder}")
            return []
        
        files = []
        for f in os.listdir(self.image_folder):
            if f.lower().endswith(('.tiff', '.tif')):
                files.append(os.path.join(self.image_folder, f))
        
        files.sort()
        print(f"📁 Found {len(files)} TIFF files")
        for f in files:
            print(f"   • {os.path.basename(f)}")
        
        return files
    
    async def upload_file(self, file_path: str) -> bool:
        """Upload a TIFF file to the web app."""
        print(f"\n📤 Uploading: {os.path.basename(file_path)}")
        try:
            file_input = await self.page.query_selector("input#fileInput")
            if not file_input:
                print("❌ File input element not found")
                return False
            
            await file_input.set_input_files(file_path)
            
            # Wait for file to load
            await self.page.wait_for_timeout(2000)
            print("✅ File uploaded")
            return True
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    async def process_image(self) -> bool:
        """Click the process button and wait for results."""
        print("⚙️  Processing image...")
        try:
            # Click process button
            process_btn = await self.page.query_selector(".process-btn")
            if not process_btn:
                print("❌ Process button not found")
                return False
            
            await process_btn.click()
            
            # Wait for processing and results to appear
            await self.page.wait_for_timeout(5000)
            
            # Check if results section appeared
            try:
                await self.page.wait_for_selector(".results.active", timeout=30000)
                print("✅ Processing complete")
                return True
            except:
                print("⚠️  Results may not have fully loaded")
                return True
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return False
    
    async def extract_miu_values(self, filename: str) -> Optional[Dict]:
        """Extract MIU and MIU STD values from the results page."""
        print("📊 Extracting MIU values...")
        try:
            result = {
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "upper_mu": None,
                "lower_mu": None,
                "upper_mu_std": None,
                "lower_mu_std": None,
            }
            
            # Wait a moment for rendering
            await self.page.wait_for_timeout(1000)
            
            # Extract from the visible text on the page
            page_content = await self.page.content()
            
            # Extract upper and lower μ values with more flexible regex
            upper_match = re.search(r'(?:Upper sample|upper)[:\s]*=\s*([\d.]+)\s*m', page_content, re.IGNORECASE)
            lower_match = re.search(r'(?:Lower sample|lower)[:\s]*=\s*([\d.]+)\s*m', page_content, re.IGNORECASE)
            
            if upper_match:
                result["upper_mu"] = float(upper_match.group(1))
            if lower_match:
                result["lower_mu"] = float(lower_match.group(1))
            
            # Try alternative extraction method - look for specific values in displayed text
            if not result["upper_mu"] and not result["lower_mu"]:
                # Try to extract from the attenuation-comparison section text
                comp_section = await self.page.text_content(".attenuation-comparison")
                if comp_section:
                    # Look for patterns like "μ_upper = 0.12345"
                    mu_upper = re.search(r'μ_upper\s*=\s*([\d.]+)', comp_section)
                    mu_lower = re.search(r'μ_lower\s*=\s*([\d.]+)', comp_section)
                    
                    if mu_upper:
                        result["upper_mu"] = float(mu_upper.group(1))
                    if mu_lower:
                        result["lower_mu"] = float(mu_lower.group(1))
            
            # Try to extract from the stats tables
            try:
                stats_rows = await self.page.query_selector_all(".stats-table tbody tr")
                for row in stats_rows:
                    cells = await row.query_selector_all("td")
                    if len(cells) >= 2:
                        label = await cells[0].text_content()
                        value = await cells[1].text_content()
                        
                        label_lower = label.strip().lower()
                        value_clean = value.strip()
                        try:
                            val = float(value_clean.replace('%', '').strip())
                            if "std" in label_lower:
                                if "upper" in label_lower:
                                    result["upper_mu_std"] = val
                                elif "lower" in label_lower:
                                    result["lower_mu_std"] = val
                        except (ValueError, AttributeError):
                            pass
            except:
                pass
            
            # Format output
            upper_mu_str = f"{result['upper_mu']:.5f}" if result['upper_mu'] else "N/A"
            lower_mu_str = f"{result['lower_mu']:.5f}" if result['lower_mu'] else "N/A"
            upper_std_str = f"{result['upper_mu_std']:.5f}" if result['upper_mu_std'] else "N/A"
            lower_std_str = f"{result['lower_mu_std']:.5f}" if result['lower_mu_std'] else "N/A"
            
            print(f"   Upper MIU: {upper_mu_str}")
            print(f"   Lower MIU: {lower_mu_str}")
            print(f"   Upper STD: {upper_std_str}")
            print(f"   Lower STD: {lower_std_str}")
            
            return result
        except Exception as e:
            print(f"❌ Failed to extract MIU values: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def export_pdf(self, filename: str) -> bool:
        """Download the PDF analysis report from the web app."""
        print(f"📄 Downloading PDF report...")
        try:
            # Look for the block PDF export button (most common for block mode)
            pdf_btn = await self.page.query_selector("#exportBlockPdfBtn")
            
            # If block button not found, try circle PDF button
            if not pdf_btn:
                pdf_btn = await self.page.query_selector("#exportCirclePdfBtn")
            
            # If still not found, search by class and text content
            if not pdf_btn:
                buttons = await self.page.query_selector_all(".export-btn")
                for btn in buttons:
                    text = await btn.text_content()
                    if "PDF" in text.upper():
                        pdf_btn = btn
                        break
            
            if not pdf_btn:
                print("⚠️  PDF export button not found on page")
                return False
            
            # Set up download handler and click button
            async with self.page.expect_download() as download_info:
                await pdf_btn.click()
                download = await download_info.value
                
                # Create safe filename (remove special characters)
                base_name = os.path.splitext(filename)[0]
                safe_filename = re.sub(r'[^\w\s-]', '', base_name)
                safe_filename = re.sub(r'[-\s]+', '_', safe_filename).strip('_')
                
                pdf_path = os.path.join(self.downloads_folder, f"{safe_filename}.pdf")
                
                # Save the downloaded PDF
                await download.save_as(pdf_path)
                print(f"✅ PDF downloaded: {safe_filename}.pdf")
                return True
                
        except Exception as e:
            print(f"⚠️  PDF download failed: {e}")
            return False
    
    async def wait_for_next_upload(self, delay: int = 2) -> None:
        """Wait before processing the next file."""
        await self.page.wait_for_timeout(delay * 1000)
    
    async def process_all_images(self) -> None:
        """Process all TIFF images in the folder."""
        files = self.get_tiff_files()
        
        if not files:
            print("❌ No TIFF files found in folder")
            return
        
        total = len(files)
        for idx, file_path in enumerate(files, 1):
            filename = os.path.basename(file_path)
            print(f"\n{'='*60}")
            print(f"Processing {idx}/{total}: {filename}")
            print(f"{'='*60}")
            
            try:
                # Upload file
                if not await self.upload_file(file_path):
                    print(f"⚠️  Skipping {filename} - upload failed")
                    continue
                
                # Process image
                if not await self.process_image():
                    print(f"⚠️  Skipping {filename} - processing failed")
                    continue
                
                # Extract MIU values
                result = await self.extract_miu_values(filename)
                if result:
                    self.miu_results.append(result)
                
                # Export PDF
                await self.export_pdf(filename)
                
                # Wait before next upload
                if idx < total:
                    await self.wait_for_next_upload(2)
                    
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue
    
    def save_csv_results(self) -> None:
        """Save MIU results to CSV file."""
        if not self.miu_results:
            print("❌ No results to save")
            return
        
        csv_path = os.path.join(self.downloads_folder, "MIU_Results.csv")
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "Filename",
                    "Upper MIU",
                    "Upper MIU STD",
                    "Lower MIU",
                    "Lower MIU STD",
                    "Timestamp"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.miu_results:
                    writer.writerow({
                        "Filename": result["filename"],
                        "Upper MIU": result["upper_mu"],
                        "Upper MIU STD": result["upper_mu_std"],
                        "Lower MIU": result["lower_mu"],
                        "Lower MIU STD": result["lower_mu_std"],
                        "Timestamp": result["timestamp"]
                    })
            
            print(f"✅ CSV saved: {csv_path}")
            print(f"📊 Processed {len(self.miu_results)} images successfully")
        except Exception as e:
            print(f"❌ Failed to save CSV: {e}")
    
    async def run(self) -> None:
        """Run the complete automation workflow."""
        try:
            print("\n" + "="*60)
            print("🎯 MIU Batch Processor - Playwright Automation")
            print("="*60)
            
            # Setup
            self.setup_directories()
            self.start_server()
            await self.init_browser()
            await self.open_app()
            
            # Process all images
            await self.process_all_images()
            
            # Save results
            self.save_csv_results()
            
            # Summary
            print("\n" + "="*60)
            print("✅ Automation Complete!")
            print("="*60)
            print(f"📊 Results location: {self.downloads_folder}")
            print(f"� Data exported:")
            print(f"   ✓ MIU_Results.csv (MIU & MIU STD values)")
            print(f"   ✓ Individual PDF reports for each image")
            print(f"📋 Total processed: {len(self.miu_results)} images")
            
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
        finally:
            await self.close_browser()
            self.stop_server()


async def main():
    """Main entry point."""
    processor = MIUProcessor()
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())
