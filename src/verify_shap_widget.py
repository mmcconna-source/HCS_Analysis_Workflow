import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.shap_widget import SHAPWidget

def test_shap_widget():
    print("Generating synthetic data...")
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some metadata columns to test filtering
    X_df['Metadata_Well'] = ['A01'] * 100
    X_df['Image_Path'] = ['/path/to/image.tif'] * 100
    
    X_train, X_test, y_train, y_test = train_test_split(X_df[feature_names], y, test_size=0.2, random_state=42)
    
    print("Training dummy XGBoost model...")
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Save model to test file loading path
    model_path = 'test_xgb_model.json'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    print("Initializing SHAPWidget with model object...")
    try:
        widget_obj = SHAPWidget(model, X_test)
        print("Widget initialized successfully with model object.")
    except Exception as e:
        print(f"Error initializing widget with object: {e}")
        
    print("Initializing SHAPWidget with model path...")
    try:
        widget_path = SHAPWidget(model_path, X_df) # Pass full DF with metadata to test filtering
        print("Widget initialized successfully with model path.")
        
        # Simulate button click (run_analysis)
        print("Simulating 'Run Analysis' click...")
        widget_path.run_analysis(None)
        
        # Check internal state
        if widget_path.shap_values is not None:
             print("SHAP values calculated successfully.")
        else:
             print("Error: SHAP values not returned.")
             
    except Exception as e:
        print(f"Error validating widget path logic: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Cleanup complete.")

if __name__ == "__main__":
    test_shap_widget()
