import sys
import os
import shutil
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

from cp_merger_ui import CPMergerUI, process_well

def create_dummy_csv(path, obj_count=5, type="Cell"):
    data = {
        "ImageNumber": [1] * obj_count,
        "ObjectNumber": list(range(1, obj_count + 1)),
        f"Intensity_Mean_{type}": [0.5] * obj_count
    }
    if type == "Cell":
        data["Metadata_Site"] = [1] * obj_count
        
    pd.DataFrame(data).to_csv(path, index=False)

def test_cp_merger():
    print("Testing CPMergerUI Logic...")
    
    base_dir = Path("test_cp_data")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()
    
    # 1. Setup Dummy Wells
    wells = ["Well_A01", "Well_A02"]
    for w in wells:
        wd = base_dir / w
        wd.mkdir()
        create_dummy_csv(wd / "MyExpt_Cell.csv", type="Cell")
        create_dummy_csv(wd / "MyExpt_Cytoplasm.csv", type="Cytoplasm")
        create_dummy_csv(wd / "MyExpt_Nucleus.csv", type="Nucleus")
        # Image CSV
        pd.DataFrame({"ImageNumber": [1], "Metadata_Site": [1]}).to_csv(wd / "MyExpt_Image.csv", index=False)
        
    print(f"  ✅ Created dummy data in {base_dir}")
    
    # 2. Test Processing Logic (Direct Call)
    print("  Testing process_well()...")
    config = {
        'cell': "MyExpt_Cell.csv",
        'cyto': "MyExpt_Cytoplasm.csv",
        'nuc': "MyExpt_Nucleus.csv",
        'img': "MyExpt_Image.csv"
    }
    
    sc_df, img_df = process_well(base_dir / "Well_A01", config)
    
    assert sc_df is not None
    assert len(sc_df) == 5
    assert "Cell_Intensity_Mean_Cell" in sc_df.columns
    assert "Cytoplasm_Intensity_Mean_Cytoplasm" in sc_df.columns
    assert "Metadata_WellID" in sc_df.columns
    assert sc_df.iloc[0]["Metadata_WellID"] == "Well_A01"
    
    print("  ✅ Single Well Merge Success")
    
    # 3. Test Full UI Logic (Mocking UI interaction by calling _run_merge logic manually or just trusting verify logic above)
    # Since _run_merge relies on Widgets, we might just stop here or instantiate UI and set values.
    
    ui = CPMergerUI()
    ui.txt_input.value = str(base_dir)
    output_dir = base_dir / "Results"
    ui.txt_output.value = str(output_dir)
    ui.chk_parallel.value = False # Test sequential first
    
    # We can't easily click button, but we can call _run_merge(None)
    print("  Testing Full UI Execution...")
    ui._run_merge(None)
    
    assert (output_dir / "master_single_cell.csv").exists()
    assert (output_dir / "master_image_level.csv").exists()
    
    master_sc = pd.read_csv(output_dir / "master_single_cell.csv")
    assert len(master_sc) == 10 # 2 wells * 5 objs
    
    print("  ✅ Full Merge Success")
    
    # Cleanup
    # shutil.rmtree(base_dir)

if __name__ == "__main__":
    test_cp_merger()
