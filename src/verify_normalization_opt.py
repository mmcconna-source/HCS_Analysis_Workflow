import sys
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# Add src to path
sys.path.append('src')

from normalization_widget import NormalizationWidget

def test_normalization_optimization():
    print("Testing NormalizationWidget Optimization (Float32)...")
    
    # 1. Setup Data with Float64
    data = {
        'Metadata_PlateID': ['P1', 'P1', 'P1'],
        'Metadata_CMPD': ['DMSO', 'DMSO', 'DrugA'],
        'Cell_FeatA': [10.0, 12.0, 55.0],  # Default float64
        'Cell_FeatB': [1.0, 1.0, 5.0]
    }
    df = pd.DataFrame(data)
    
    # Verify input is float64
    print(f"Input Dtype: {df['Cell_FeatA'].dtype}")
    assert df['Cell_FeatA'].dtype == 'float64', "Input should be float64"
    
    # 2. Test Loading (Should Downcast)
    widget = NormalizationWidget()
    widget.load_data(df)
    
    # Check internal df
    internal_dtype = widget.input_df['Cell_FeatA'].dtype
    print(f"Loaded Internal Dtype: {internal_dtype}")
    assert internal_dtype == 'float32', "Widget should downcast input to float32 on load"
    
    # 3. Test Normalization (Output should be Float32)
    # Mock Selection
    widget.dd_plate_col.value = 'Metadata_PlateID'
    widget.dd_control_col.value = 'Metadata_CMPD'
    widget.txt_control_val.value = 'DMSO'
    widget.dd_method.value = 'Percentage of Control (Mean)'
    
    # Run
    widget._run_normalization(None)
    
    norm_df = widget.normalized_df
    result_dtype = norm_df['Cell_FeatA'].dtype
    print(f"Output Normalization Dtype: {result_dtype}")
    assert result_dtype == 'float32', "Normalized output should be float32"
    
    print("✅ Normalization Optimization Verification Passed")

if __name__ == "__main__":
    test_normalization_optimization()
