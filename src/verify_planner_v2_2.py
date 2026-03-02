import sys
import pandas as pd
from pathlib import Path
import os

# Add src to path
sys.path.append('src')

from experiment_planning_ui import InteractivePlatePlanner

def test_planner_v2_2():
    print("Testing InteractivePlatePlanner V2.2 (CSV Import)...")
    
    # 1. Setup Dummy CSV
    csv_path = "test_library.csv"
    with open(csv_path, 'w') as f:
        f.write("CompoundName,Mechanism,Batch\n")
        f.write("TestDrugA,Inhibitor,001\n")
        f.write("TestDrugB,Activator,002\n")
    
    try:
        planner = InteractivePlatePlanner(plate_type=96)
        
        # 2. Perform Import
        planner.txt_import_path.value = csv_path
        planner._import_csv_lib(None)
        
        # 3. Verify
        assert "TestDrugA" in planner.compounds
        assert planner.compounds['TestDrugA']['meta']['Mechanism'] == "Inhibitor"
        assert planner.compounds['TestDrugA']['meta']['Batch'] == "001"
        
        assert "TestDrugB" in planner.compounds
        assert planner.compounds['TestDrugB']['meta']['Mechanism'] == "Activator"
        
        print("CSV Import Logic: SUCCESS")
        
    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)

    print("\nALL V2.2 CHECKS PASSED")

if __name__ == "__main__":
    test_planner_v2_2()
