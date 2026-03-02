import sys
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# Add src to path
sys.path.append('src')

from normalization_widget import NormalizationWidget

def test_normalization():
    print("Testing NormalizationWidget Logic...")
    
    # 1. Setup Data: 2 Plates, 2 Features
    # Plate 1: DMSO Mean = 10, Cell_A needs norm
    # Plate 2: DMSO Mean = 20, Cell_A needs norm
    
    data = {
        'Metadata_PlateID': ['P1', 'P1', 'P1', 'P2', 'P2', 'P2'],
        'Metadata_CMPD': ['DMSO', 'DMSO', 'DrugA', 'DMSO', 'DMSO', 'DrugA'],
        'Cell_FeatA': [9.0, 11.0, 50.0, 19.0, 21.0, 100.0], # Means: P1=10, P2=20
        'Cell_FeatB': [1.0, 1.0, 5.0, 2.0, 2.0, 10.0]
    }
    df = pd.DataFrame(data)
    
    # 2. Test
    widget = NormalizationWidget()
    widget.load_data(df)
    
    # Mock Selection
    widget.dd_plate_col.value = 'Metadata_PlateID'
    widget.dd_control_col.value = 'Metadata_CMPD'
    widget.txt_control_val.value = 'DMSO'
    widget.dd_method.value = 'Percentage of Control (Mean)'
    
    # Run
    widget._run_normalization(None)
    
    norm_df = widget.normalized_df
    assert norm_df is not None, "Normalized DF should not be None"
    
    # Verify P1 (Mean=10) -> DrugA(50) should become 500%
    p1_drug = norm_df[ (norm_df['Metadata_PlateID']=='P1') & (norm_df['Metadata_CMPD']=='DrugA') ]
    val_a = p1_drug['Cell_FeatA'].values[0]
    if not np.isclose(val_a, 500.0):
        print(f"❌ P1 Feature A Expected 500.0, got {val_a}")
    else:
        print("✅ P1 Normalization Correct")
        
    # Verify P2 (Mean=20) -> DrugA(100) should become 500%
    p2_drug = norm_df[ (norm_df['Metadata_PlateID']=='P2') & (norm_df['Metadata_CMPD']=='DrugA') ]
    val_a_2 = p2_drug['Cell_FeatA'].values[0]
    if not np.isclose(val_a_2, 500.0):
        print(f"❌ P2 Feature A Expected 500.0, got {val_a_2}")
    else:
        print("✅ P2 Normalization Correct")
        
    print(f"Normalized DF Shape: {norm_df.shape}")

if __name__ == "__main__":
    test_normalization()
