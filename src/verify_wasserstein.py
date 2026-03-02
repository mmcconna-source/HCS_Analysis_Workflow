import pandas as pd
import numpy as np
from src.wasserstein_widget import WassersteinDistanceWidget
import matplotlib.pyplot as plt
import io

def verify_wasserstein():
    # 1. Create Dummy Data
    np.random.seed(42)
    n_samples = 100
    
    # Generate data with known shifts
    # Group A: Baseline (e.g. DMSO)
    group_a = pd.DataFrame({
        'Metadata_Compound': ['DMSO'] * n_samples,
        'Feature_1': np.random.normal(0, 1, n_samples), # Standard Normal
        'Feature_2': np.random.normal(5, 1, n_samples)  # Mean 5
    })
    
    # Group B: Shifted Feature 1 (e.g. Drug1)
    group_b = pd.DataFrame({
        'Metadata_Compound': ['Drug1'] * n_samples,
        'Feature_1': np.random.normal(2, 1, n_samples), # Shifted Mean +2
        'Feature_2': np.random.normal(5, 1, n_samples)  # Same Mean 5
    })
    
    # Group C: Shifted Feature 2 (e.g. Drug2)
    group_c = pd.DataFrame({
        'Metadata_Compound': ['Drug2'] * n_samples,
        'Feature_1': np.random.normal(0, 1, n_samples), # Same as DMSO
        'Feature_2': np.random.normal(8, 1, n_samples)  # Shifted Mean +3
    })
    
    df = pd.concat([group_a, group_b, group_c], ignore_index=True)
    
    print("Dummy Data Created.")
    print(df.groupby('Metadata_Compound').mean())
    
    # 2. Initialize Widget
    widget = WassersteinDistanceWidget(df)
    
    # 3. Configure Widget Programmatically
    widget.group_col_dropdown.value = 'Metadata_Compound'
    widget.reference_dropdown.value = 'DMSO'
    widget.test_group_select.value = ('Drug1', 'Drug2')
    
    print("\nRunning Analysis...")
    widget.run_analysis(None)
    
    # 4. Verify Results
    results = widget.results_df
    if results is None or results.empty:
        print("Test FAILED: Results DataFrame is empty.")
    dist_d1_f1 = results.loc['Feature_1', 'Drug1']
    dist_d1_f2 = results.loc['Feature_2', 'Drug1']
    dist_d2_f1 = results.loc['Feature_1', 'Drug2']
    dist_d2_f2 = results.loc['Feature_2', 'Drug2']

    print(f"\nDistance Drug1 Feature_1 (Expected ~2.0): {dist_d1_f1:.2f}")
    print(f"Distance Drug1 Feature_2 (Expected ~0.0): {dist_d1_f2:.2f}")
    print(f"Distance Drug2 Feature_1 (Expected ~0.0): {dist_d2_f1:.2f}")
    print(f"Distance Drug2 Feature_2 (Expected ~3.0): {dist_d2_f2:.2f}")

    assert dist_d1_f1 > 1.5, "Drug1 Feature_1 distance too low"
    assert dist_d1_f2 < 0.5, "Drug1 Feature_2 distance too high"
    assert dist_d2_f1 < 0.5, "Drug2 Feature_1 distance too high"
    assert dist_d2_f2 > 2.5, "Drug2 Feature_2 distance too low"

    print("\nTest PASSED: Single-col analysis successful and distances match expectations.")

    # 4. Verify Matrix Mode
    print("\nTesting 'All vs All' Mode...")
    widget.mode_toggle.value = 'All vs All'
    widget.group_cols_select.value = ('Metadata_Treatment',)
    widget.test_group_select.value = ('DMSO', 'Drug1', 'Drug2') 
    
    widget.run_analysis(None)
    
    matrix = widget.results_df
    assert matrix.shape == (3, 3)
    # Check diagonal is 0
    assert matrix.loc['DMSO', 'DMSO'] == 0.0, "Diagonal should be 0"
    # Check symmetry
    assert np.isclose(matrix.loc['DMSO', 'Drug1'], matrix.loc['Drug1', 'DMSO']), "Matrix should be symmetric"
    print("Test PASSED: All-vs-All Matrix analysis successful.")

    # 5. Verify Multi-Column Grouping
    print("\nTesting Multi-Column Grouping (Treatment + Dose)...")
    widget.mode_toggle.value = 'Reference vs All'
    widget.group_cols_select.value = ('Metadata_Treatment', 'Metadata_Dose')
    
    # Check if options are updated correctly
    print("Available Groups:", widget.reference_dropdown.options)
    # Expected: DMSO_0.1, DMSO_1.0, Drug1_0.1, ...
    
    # Select Reference
    ref_val = sorted(widget.reference_dropdown.options)[0] # Likely DMSO_0.1
    widget.reference_dropdown.value = ref_val
    
    # Select all others as test
    all_opts = list(widget.reference_dropdown.options)
    test_vals = [x for x in all_opts if x != ref_val]
    widget.test_group_select.value = tuple(test_vals)
    
    widget.run_analysis(None)
    
    assert widget.results_df is not None
    assert not widget.results_df.empty
    print("Multi-column Results Shape:", widget.results_df.shape)
    print(widget.results_df.head())
    
    print("Test PASSED: Multi-column analysis successful.")

if __name__ == "__main__":
    verify_wasserstein()
