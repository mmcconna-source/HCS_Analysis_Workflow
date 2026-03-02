import sys
from pathlib import Path
import matplotlib.colors as mcolors

# Add src to path
sys.path.append('src')

from experiment_planning_ui import InteractivePlatePlanner

def test_color_logic():
    print("Testing InteractivePlatePlanner Coloring Logic...")
    
    planner = InteractivePlatePlanner(plate_type=96)
    
    # 1. Setup Data
    planner.well_data['A01'] = {'Drug': 'A', 'Dose': 100}
    planner.well_data['A02'] = {'Drug': 'A', 'Dose': 10}
    planner.well_data['B01'] = {'Drug': 'B', 'Dose': 100}
    
    # 2. Test Key Discovery
    planner._update_known_keys()
    assert 'Drug' in planner.known_keys
    assert 'Dose' in planner.known_keys
    print("1. Key Discovery: SUCCESS")
    
    # 3. Test Color Mapping (Categorical)
    planner.dropdown_color_by.value = 'Drug'
    planner._determine_color_map()
    
    color_a = planner.category_colors.get('A')
    color_b = planner.category_colors.get('B')
    
    assert color_a is not None
    assert color_b is not None
    assert color_a != color_b
    print(f"2. Categorical Coloring: SUCCESS (A={color_a}, B={color_b})")
    
    # 4. Test Opacity (Numeric)
    planner.dropdown_opacity_by.value = 'Dose'
    
    # Recalculate colors manually checking logic or simulating _refresh_visuals
    # We can inspect the private method _get_well_color if we refactored, 
    # but the logic is embedded in _refresh_visuals in the current implementation 
    # to directly update widgets.
    # Let's inspect the logic helper _get_opacity_map
    
    op_map = planner._get_opacity_map()
    assert op_map['min'] == 10
    assert op_map['max'] == 100
    print(f"3. Opacity Range Detected: {op_map}")
    
    # 5. Logic Check for Alpha
    # A01 (Dose 100) should be max opacity (1.0)
    # A02 (Dose 10) should be min opacity (0.3)
    
    # We can't easily check button styles without a running widget backend in this script,
    # but we can verify no errors were thrown during logic execution.
    
    try:
        planner._refresh_visuals()
        print("4. Visual Refresh Execution: SUCCESS (No Errors)")
    except Exception as e:
        print(f"4. Visual Refresh FAILED: {e}")
        raise e

    print("\nColor Verification Complete.")

if __name__ == "__main__":
    test_color_logic()
