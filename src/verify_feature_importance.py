
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')
from feature_importance_widget import FeatureImportanceWidget

def test_feature_importance():
    print("Creating dummy data...")
    # Create 100 cells, 10 features
    # Group A: "Cluster 1" -> Feature_Signal is High (100 +/- 10)
    # Group B: "Cluster 2" -> Feature_Signal is Low (10 +/- 10)
    # Noise features: Random
    
    n = 200
    df = pd.DataFrame({
        'leiden': ['1'] * 100 + ['2'] * 100,
        'Feature_Signal': np.concatenate([
            np.random.normal(100, 10, 100),
            np.random.normal(10, 10, 100)
        ]),
        'Feature_Noise1': np.random.normal(50, 50, 200),
        'Feature_Noise2': np.random.normal(0, 1, 200),
        'UMAP1': np.random.rand(200),
        'UMAP2': np.random.rand(200)
    })
    
    print("Initializing Widget...")
    widget = FeatureImportanceWidget(df)
    
    # Simulate UI Selection
    widget.group_col_dropdown.value = 'leiden'
    widget.group_a_select.value = ('1',)
    widget.group_b_select.value = ('2',)
    
    print("Running Analysis (Random Forest)...")
    widget.run_analysis(None)
    
    # Verify Results 1
    _verify_result(widget, "Random Forest")
    
    # Test XGBoost
    print("\nSwitching to XGBoost...")
    widget.model_dropdown.value = 'XGBoost'
    
    try:
        import xgboost
        widget.run_analysis(None)
        _verify_result(widget, "XGBoost")
    except ImportError:
        print("Skipping XGBoost test (not installed).")

def _verify_result(widget, model_name):
    if widget.results_df is not None and not widget.results_df.empty:
        top_feature = widget.results_df.iloc[0]['Feature']
        importance = widget.results_df.iloc[0]['Importance']
        print(f"[{model_name}] Top Feature: {top_feature} (Importance: {importance:.4f})")
        
        if top_feature == 'Feature_Signal':
            print(f"SUCCESS: {model_name} correctly identified the signal feature.")
        else:
            print(f"FAILURE: {model_name} failed. Top was {top_feature}.")
    else:
        print(f"FAILURE: {model_name} produced no results.")

if __name__ == "__main__":
    test_feature_importance()
