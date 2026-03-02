import sys
import os
import shutil
import numpy as np
from PIL import Image

# Add src to path
sys.path.append('src')

from image_merging_ui import ImageMergerUI

def create_dummy_tiff(path, shape=(32, 32)):
    # Create a simple gradient or random noise
    data = np.random.randint(0, 255, shape, dtype=np.uint8)
    img = Image.fromarray(data)
    img.save(path)

def test_merger():
    print("Testing ImageMergerUI Logic...")
    
    base_dir = "test_merger_data"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    # 1. Setup Test Data (One folder per type to avoid regex collision? 
    # The UI assumes one input folder. So let's test sequentially.)
    
    # --- Test Case A: CV8000 ---
    print("\n[Test Case A: CV8000]")
    dir_a = os.path.join(base_dir, "CV8000")
    os.makedirs(dir_a)
    # Filename: MMC0015_B07_T0001F001L01A01Z01C01.tiff
    names_a = [
        "MMC0015_B07_T0001F001L01A01Z01C01.tiff", # Chan 1
        "MMC0015_B07_T0001F001L01A01Z01C02.tiff", # Chan 2
        "MMC0015_B07_T0001F001L01A01Z01C03.tiff", # Chan 3
        # Another Field
        "MMC0015_B07_T0001F002L01A01Z01C01.tiff", 
    ]
    for n in names_a: create_dummy_tiff(os.path.join(dir_a, n))
    
    ui = ImageMergerUI()
    ui.txt_input.value = dir_a
    ui.dd_type.value = 'CV8000'
    ui._scan_channels(None)
    
    detected = set(ui.detected_channels)
    expected = {'C1', 'C2', 'C3'}
    assert detected == expected, f"CV8000 Failed: Got {detected}, expected {expected}"
    print("  ✅ Channel Detection OK")

    # --- Test Case B: CV8000_Stitched ---
    print("\n[Test Case B: CV8000_Stitched]")
    dir_b = os.path.join(base_dir, "Stitched")
    os.makedirs(dir_b)
    # Filename: B-02_F0001_T0001_Z0001_C01.tiff
    names_b = [
        "B-02_F0001_T0001_Z0001_C01.tiff",
        "B-02_F0001_T0001_Z0001_C04.tiff"
    ]
    for n in names_b: create_dummy_tiff(os.path.join(dir_b, n))
    
    ui.txt_input.value = dir_b
    ui.dd_type.value = 'CV8000_Stitched'
    ui._scan_channels(None)
    
    detected = set(ui.detected_channels)
    expected = {'C1', 'C4'}
    assert detected == expected, f"Stitched Failed: Got {detected}, expected {expected}"
    print("  ✅ Channel Detection OK")

    # --- Test Case C: CQ1 ---
    print("\n[Test Case C: CQ1]")
    dir_c = os.path.join(base_dir, "CQ1")
    os.makedirs(dir_c)
    # Filename: W0014F0001T0001Z000C1.tiff
    names_c = [
        "W0014F0001T0001Z000C1.tiff",
        "W0014F0001T0001Z000C2.tiff"
    ]
    for n in names_c: create_dummy_tiff(os.path.join(dir_c, n))
    
    ui.txt_input.value = dir_c
    ui.dd_type.value = 'CQ1'
    ui._scan_channels(None)
    
    detected = set(ui.detected_channels)
    expected = {'C1', 'C2'}
    assert detected == expected, f"CQ1 Failed: Got {detected}, expected {expected}"
    print("  ✅ Channel Detection OK")
    
    # Clean up
    # shutil.rmtree(base_dir) # Keep for inspection if needed, or remove
    print("\nALL SCANS PASSED.")

if __name__ == "__main__":
    test_merger()
