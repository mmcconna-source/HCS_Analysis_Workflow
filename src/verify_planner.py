import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

from experiment_planning_ui import InteractivePlatePlanner

def test_interactive_planner():
    print("Testing InteractivePlatePlanner Logic...")
    
    # 1. Initialize
    planner = InteractivePlatePlanner(plate_type=96)
    print("1. Initialization: SUCCESS")
    
    # 2. Simulate Selection
    # Select A01, A02, A03
    planner.selected_wells = {'A01', 'A02', 'A03'}
    print(f"2. Selection simulated: {planner.selected_wells}")
    
    # 3. Test Assignment
    planner.input_key.value = 'Drug'
    planner.input_value.value = 'TestDrug'
    planner._assign_condition(None) # Simulate button click
    
    # Verify
    assert planner.well_data['A01']['Drug'] == 'TestDrug'
    assert planner.well_data['A02']['Drug'] == 'TestDrug'
    print("3. Condition Assignment: SUCCESS")
    
    # 4. Test Dilution
    # Select Row B: B01 to B04
    planner.selected_wells = {'B01', 'B02', 'B03', 'B04'}
    planner.input_key.value = 'Concentration'
    planner.dilution_start.value = 100
    planner.dilution_factor.value = 2
    planner.dilution_steps.value = 4
    planner.dilution_replicates.value = 1
    planner._generate_dilution(None)
    
    # Verify: 100, 50, 25, 12.5
    assert planner.well_data['B01']['Concentration'] == 100.0
    assert planner.well_data['B02']['Concentration'] == 50.0
    assert planner.well_data['B03']['Concentration'] == 25.0
    assert planner.well_data['B04']['Concentration'] == 12.5
    print("4. Dilution Series: SUCCESS")
    
    # 5. Test Randomization
    # Create distinct data in C01 and C02
    planner.well_data['C01'] = {'ID': 1}
    planner.well_data['C02'] = {'ID': 2}
    planner.selected_wells = {'C01', 'C02'}
    
    # Run randomization multiple times to ensure it can shuffle (probabilistic)
    # We check that the set of values remains the same
    start_values = [planner.well_data['C01']['ID'], planner.well_data['C02']['ID']]
    planner._randomize(None)
    end_values = []
    if 'C01' in planner.well_data: end_values.append(planner.well_data['C01']['ID'])
    if 'C02' in planner.well_data: end_values.append(planner.well_data['C02']['ID'])
    
    assert sorted(start_values) == sorted(end_values)
    print("5. Randomization Data Integrity: SUCCESS")
    
    # 6. Test Export
    test_csv = Path("test_export.csv")
    planner.output_path.value = str(test_csv)
    planner._export_csv(None)
    
    assert test_csv.exists()
    df = pd.read_csv(test_csv)
    # Check if A01 Drug exists
    row_a01 = df[df['WellID'] == 'A01'].iloc[0]
    assert row_a01['Drug'] == 'TestDrug'
    
    # Check B01 Concentration
    row_b01 = df[df['WellID'] == 'B01'].iloc[0]
    assert row_b01['Concentration'] == 100.0
    
    print("6. CSV Export: SUCCESS")
    
    # Clean up
    if test_csv.exists():
        test_csv.unlink()

if __name__ == "__main__":
    try:
        test_interactive_planner()
        print("\nALL CHECKS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
