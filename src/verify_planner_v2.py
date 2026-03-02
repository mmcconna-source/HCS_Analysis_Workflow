import sys
from pathlib import Path
import random

# Add src to path
sys.path.append('src')

from experiment_planning_ui import InteractivePlatePlanner

def test_planner_v2():
    print("Testing InteractivePlatePlanner V2 (Compound-Centric)...")
    
    planner = InteractivePlatePlanner(plate_type=96)
    
    # 1. Test Compound Manager
    planner.txt_compound_name.value = "Taxol"
    planner._add_compound(None)
    
    assert "Taxol" in planner.compounds
    assert planner.active_compound == "Taxol" # Should auto-select
    planner.dropdown_compounds.value = "Taxol" # Explicit set
    
    print("1. Compound Manager: SUCCESS")
    
    # 2. Test Manual Assignment
    # Select A01-A03
    planner.selected_wells = {'A01', 'A02', 'A03'}
    planner.input_manual_dose.value = 5.0
    planner._assign_manual(None)
    
    assert planner.well_data['A01']['Metadata_Drug'] == "Taxol"
    assert planner.well_data['A01']['Metadata_Dose'] == 5.0
    print("2. Manual Assignment: SUCCESS")
    
    # 3. Test Dilution (Fill Logic)
    # Start at B01. Horizontal. 4 Steps, 2 Reps.
    # Should fill B01-B02 (Step 1), B03-B04 (Step 2)...
    planner.selected_wells = {'B01'} # Only 1 well selected as start
    planner.dil_start.value = 100
    planner.dil_factor.value = 10
    planner.dil_steps.value = 3
    planner.dil_reps.value = 2
    planner.dil_dir.value = 'Horizontal'
    
    planner._generate_dilution(None)
    
    # Check Step 1 (100)
    assert planner.well_data['B01']['Metadata_Dose'] == 100.0
    assert planner.well_data['B02']['Metadata_Dose'] == 100.0
    
    # Check Step 2 (10)
    assert planner.well_data['B03']['Metadata_Dose'] == 10.0
    assert planner.well_data['B04']['Metadata_Dose'] == 10.0
    
    # Check Step 3 (1)
    assert planner.well_data['B05']['Metadata_Dose'] == 1.0
    
    print("3. Dilution (Auto-Fill): SUCCESS")
    
    # 4. Test Global Randomization
    # Collect current data state
    start_data_count = len(planner.well_data)
    start_data_A01 = planner.well_data['A01'].copy()
    
    # Run randomization
    # Note: Randomization shuffles data amongst USED wells.
    # Since A01, A02... B06 are used.
    # Logic: Shuffle payloads, put back into [A01, A02... B06]
    
    # To verify shuffle happened, we need enough data points that probability of staying same is low.
    # But for unit test, just checking data integrity is safer.
    
    planner._randomize_plate(None)
    
    end_data_count = len(planner.well_data)
    assert start_data_count == end_data_count
    
    # Check data integrity (values still exist somewhere)
    found_taxol_5 = False
    for d in planner.well_data.values():
        if d.get('Metadata_Dose') == 5.0 and d.get('Metadata_Drug') == 'Taxol':
            found_taxol_5 = True
            break
            
    assert found_taxol_5
    print("4. Randomization: SUCCESS (Data Integrity Preserved)")
    
    print("\nALL V2 CHECKS PASSED")

if __name__ == "__main__":
    test_planner_v2()
