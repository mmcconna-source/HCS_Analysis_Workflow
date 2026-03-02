import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from experiment_planning_ui import InteractivePlatePlanner

def test_planner_v2_1():
    print("Testing InteractivePlatePlanner V2.1 (Fixed Metadata)...")
    
    planner = InteractivePlatePlanner(plate_type=96)
    
    # 1. Test Compound Creation with Metadata
    planner.txt_compound_name.value = "DrugX"
    planner.txt_compound_meta.value = "Mechanism=Inhibitor; Batch=B001"
    planner._add_compound(None)
    
    assert "DrugX" in planner.compounds
    assert planner.compounds['DrugX']['meta']['Mechanism'] == "Inhibitor"
    assert planner.compounds['DrugX']['meta']['Batch'] == "B001"
    
    print("1. Metadata Parsing: SUCCESS")
    
    # 2. Test Manual Assignment (Verification of Inheritance)
    planner.selected_wells = {'C01'}
    planner.input_manual_dose.value = 10.0
    planner._assign_manual(None)
    
    well_data = planner.well_data.get('C01', {})
    assert well_data.get('Metadata_Drug') == "DrugX"
    assert well_data.get('Metadata_Dose') == 10.0
    assert well_data.get('Mechanism') == "Inhibitor"
    assert well_data.get('Batch') == "B001"
    
    print("2. Manual Assignment Metadata Inheritance: SUCCESS")
    
    # 3. Test Dilution Series Metadata Inheritance
    planner.selected_wells = {'D01'} # Start well
    planner.dil_start.value = 100
    planner.dil_factor.value = 2
    planner.dil_steps.value = 2
    planner.dil_reps.value = 1
    planner._generate_dilution(None)
    
    # Check D01 (Step 1)
    w1 = planner.well_data.get('D01', {})
    assert w1.get('Metadata_Dose') == 100.0
    assert w1.get('Mechanism') == "Inhibitor"
    
    # Check D02 (Step 2)
    w2 = planner.well_data.get('D02', {})
    assert w2.get('Metadata_Dose') == 50.0
    assert w2.get('Mechanism') == "Inhibitor"
    
    print("3. Dilution Assignments Metadata Inheritance: SUCCESS")
    
    print("\nALL V2.1 CHECKS PASSED")

if __name__ == "__main__":
    test_planner_v2_1()
