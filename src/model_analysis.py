import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output
import shap
import xgboost as xgb
import os

def run_shap_analysis(model, X, class_names=None):
    """
    Run SHAP analysis on a trained XGBoost model.
    Displays summary plot (beeswarm) and feature importance bar plot.
    
    Args:
        model: Trained XGBoost model object or path to saved model file (e.g., .ubj, .model).
        X: Feature DataFrame (or subset) to explain.
        class_names: List of class names (optional).
    """

    print("Initializing SHAP explainer...")
    
    # 1. Load Model (if filename provided)
    if isinstance(model, str):
        if not os.path.exists(model):
            print(f"Error: Model file '{model}' not found.")
            return

        print(f"Loading model from file: {model}")
        try:
            # Create a fresh classifier and load the model
            # load_model handles JSON, UBJSON, and Binary formats automatically
            loaded_model = xgb.XGBClassifier()
            loaded_model.load_model(model)
            model = loaded_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    # 2. Prepare Data (X)
    if X is None or (hasattr(X, 'empty') and X.empty):
        print("Error: No data (X) provided for SHAP analysis. Please provide a DataFrame.")
        return

    # Ensure Numeric Data Only
    # Filter out any non-numeric columns that might have slipped in
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Explicitly drop Metadata/Image columns
    cols_to_keep = [c for c in X_numeric.columns if not c.startswith('Metadata_') and not c.startswith('Image_')]
    X_numeric = X_numeric[cols_to_keep]
    
    if X_numeric.shape[1] < X.shape[1]:
        dropped = set(X.columns) - set(X_numeric.columns)
        print(f"Warning: Dropped {len(dropped)} non-feature columns from SHAP data.")
        
    X = X_numeric

    if X.empty:
        print("Error: No numeric feature data remaining after filtering.")
        return

    explainer = None
    shap_values = None

    # 3. Initialize SHAP Explainer
    try:
        # Attempt 1: Standard TreeExplainer with wrapper
        print("   Attempt 1: shap.TreeExplainer(model)...")
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"   Attempt 1 failed: {e}")
        try:
             # Attempt 2: Pass booster directly
             print("   Attempt 2: shap.TreeExplainer(model.get_booster())...")
             booster = model.get_booster()
             explainer = shap.TreeExplainer(booster)
        except Exception as e2:
             print(f"   Attempt 2 failed: {e2}")
             
             try:
                 # Attempt 3: Aggressive Text Patch (Regex)
                 # XGBoost persists "base_score" as a list string in JSON, which crashes SHAP.
                 # We will dump to JSON, edit the raw text to force a scalar, and reload.
                 print("   Attempt 3: Regex Patch on JSON dump...")
                 import tempfile
                 import json
                 import re
                 
                 # Save current state to temp JSON
                 with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp:
                     tmp_path = tmp.name
                 
                 booster = model.get_booster()
                 booster.save_model(tmp_path)
                 
                 with open(tmp_path, 'r') as f:
                     content = f.read()
                     
                 # Regex to find "base_score":["..."] or "base_score":"..."
                 # We want to force it to "base_score":"0.5"
                 # Pattern matches: "base_score": followed by optional space, then either a string or a list of strings
                 # We replace it with "base_score":"0.5"
                 
                 # This pattern catches "base_score":["0.5"] or "base_score":"0.5" or "base_score":[ "0.123" ]
                 pattern = r'"base_score"\s*:\s*(\[[^\]]+\]|"[^"]+")'
                 
                 if re.search(pattern, content):
                     print("      Found 'base_score' pattern. Patching...")
                     new_content = re.sub(pattern, '"base_score":"0.5"', content)
                     
                     with open(tmp_path, 'w') as f:
                         f.write(new_content)
                         
                     # Load back into fresh model
                     patched_model = xgb.XGBClassifier()
                     patched_model.load_model(tmp_path)
                     explainer = shap.TreeExplainer(patched_model)
                     print("      Success with regex patch!")
                 else:
                     print("      'base_score' pattern not found in dump.")
                     
                 # Cleanup
                 if os.path.exists(tmp_path):
                     os.remove(tmp_path)
                     
             except Exception as e3:
                 print(f"   Attempt 3 failed: {e3}")

                 # Fallback: KernelExplainer (Model Agnostic / Slower)
                 print("   Fallback: shap.KernelExplainer (slow)...")
        try:
            # Use background summary (kmeans) for speed
            background = shap.kmeans(X, 50) if len(X) > 50 else X
            
            # Helper to ensure numeric input for predict_proba
            def predict_wrapper(data):
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data, columns=X.columns)
                return model.predict_proba(data)
                
            explainer = shap.KernelExplainer(predict_wrapper, background)
        except Exception as k_e:
            print(f"   KernelExplainer failed: {k_e}")

    # 4. Calculate Claims & Plot
    if explainer is None:
        print("Failed to initialize SHAP explainer.")
        
        # Ultimate Fallback: XGBoost's built-in importance
        try:
             print("Showing XGBoost built-in feature importance as fallback...")
             from xgboost import plot_importance
             plt.figure(figsize=(10, 6))
             plot_importance(model, max_num_features=20, height=0.5)
             plt.title("XGBoost Feature Importance (Gain)")
             plt.show()
        except:
             pass
        return

    try:
        print(f"   Calculating SHAP values...")
        # KernelExplainer needs array, TreeExplainer usually handles both
        X_for_shap = X
        if isinstance(explainer, shap.KernelExplainer):
             X_for_shap = X.values
             
        shap_values = explainer.shap_values(X_for_shap)
        
        print("   Generating SHAP plots...")
        
        # Summary Plot (Beeswarm)
        plt.figure()
        shap.summary_plot(shap_values, X, class_names=class_names, show=True)
        
        # Bar Plot (Feature Importance)
        plt.figure()
        shap.summary_plot(shap_values, X, class_names=class_names, plot_type="bar", show=True)
        
        print("SHAP analysis complete.")
        
    except Exception as e:
        print(f"Error during SHAP plotting: {e}")
        import traceback
        traceback.print_exc()
