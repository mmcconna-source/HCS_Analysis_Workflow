import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path (assuming script is run from root or src)
# We need the parent of src if we import as 'from src.mod'
sys.path.append(os.path.abspath('.'))

from src.classification_widget import ClassificationWidget
from src.tile_extraction import NotebookConfig

def test_classification_widget_workflow():
    print("Testing ClassificationWidget Workflow...")
    
    # 1. Mock Data
    # Increase sample size to ensure enough for train/test split
    n_samples = 200
    df = pd.DataFrame({
        'Feature1': np.random.rand(n_samples),
        'Feature2': np.random.rand(n_samples),
        'Metadata_WellID': ['A01'] * n_samples,
        'Metadata_Field': [1] * n_samples,
        'ObjectNumber': range(1, n_samples + 1),
        # Add UMAP for exclusion logic
        'UMAP1': np.random.rand(n_samples), 
        'UMAP2': np.random.rand(n_samples)
    })
    
    # Add some signal for classification
    # Feature 1 > 0.6 is Positive
    df['Signal'] = (df['Feature1'] > 0.6).astype(int)
    
    config = NotebookConfig(
        csv_path='test.csv',
        image_base_path='test_images',
        channel_names=['DAPI', 'GFP']
    )
    
    # 2. Initialize Widget
    widget = ClassificationWidget(df, config)
    print("   SUCCESS: Widget initialized.")
    
    # 3. Simulate Annotation
    # Manually populate annotations to skip UI interaction
    print("   Simulating manual annotations...")
    
    # Annotate 40 cells (split roughly 50/50 based on signal)
    # Ensure we get both classes
    pos_indices = df[df['Signal'] == 1].index[:20]
    neg_indices = df[df['Signal'] == 0].index[:20]
    
    for idx in pos_indices:
        widget.annotations[idx] = 'Positive'
    for idx in neg_indices:
        widget.annotations[idx] = 'Negative'
        
    print(f"   Annotated {len(widget.annotations)} cells (20 Pos, 20 Neg).")
    
    # 4. Train Model
    print("   Training model...")
    try:
        widget.test_size_slider.value = 0.2
        widget.train_model(None)
        
        if widget.model is None:
             print("   FAILURE: Model not trained.")
        else:
             print("   SUCCESS: Model trained.")
             
    except Exception as e:
        print(f"   FAILURE: Training raised exception: {e}")
        import traceback
        traceback.print_exc()

    # 5. Test Global Scaling Logic
    print("   Testing Global Scaling in ChannelMappingWidget...")
    try:
        tiles = []
        for i in range(5):
             # Channel 0: Max = (i+1)*100
             t = np.zeros((2, 10, 10), dtype=np.uint16)
             t[0] = (i + 1) * 100
             tiles.append(t)
        
        widget.channel_widget.sample_tiles = tiles
        widget.channel_widget._calculate_global_stats()
        
        stats = widget.channel_widget._global_stats.get(0)
        if stats:
             p_min, p_max, abs_max = stats
             if abs_max == 500:
                 print("   SUCCESS: Global max calculation correct.")
             else:
                 print(f"   FAILURE: Expected max 500, got {abs_max}")
                 
             widget.channel_widget.global_scale_checkbox.value = True
             slider_max = widget.channel_widget._widgets[0]['max'].max
             if slider_max > 100:
                  print(f"   SUCCESS: Slider range updated to absolute (Max={slider_max}).")
             else:
                  print("   FAILURE: Slider range did not switch to absolute.")
        else:
             print("   FAILURE: Global stats not calculated.")
             
    except Exception as e:
         print(f"   FAILURE: Global scaling test failed: {e}")

    # 6. Predict All
    print("   Predicting on full dataset (in-place)...")
    try:
        widget.predict_all(None)
        
        if 'Metadata_PredictedClass' in widget.df.columns:
             print("   SUCCESS: 'Metadata_PredictedClass' column added.")
        else:
             print("   FAILURE: 'Metadata_PredictedClass' column missing.")

    except Exception as e:
        print(f"   FAILURE: Prediction raised exception: {e}")

    # 7. Test Tile Caching
    print("   Testing Tile Caching...")
    try:
        widget.reset_annotations(None)
        if hasattr(widget, 'cached_tiles'):
             print("   SUCCESS: cached_tiles attribute exists.")
        else:
             print("   FAILURE: cached_tiles attribute missing.")
    except Exception as e:
         print(f"   FAILURE: Caching test failed: {e}")
         
    # 8. Test SHAP Analysis
    print("   Testing SHAP Analysis Helper...")
    try:
        from src.model_analysis import run_shap_analysis
        if widget.model is not None and widget.X_test is not None:
            # We mock shap if not installed, effectively? model_analysis handles import error.
            # Just run it.
            run_shap_analysis(widget.model, widget.X_test, widget.class_names)
            print("   SUCCESS: run_shap_analysis executed (check for output errors).")
        else:
            print("   SKIPPED: Model not trained/available for SHAP test.")
            
    except Exception as e:
        print(f"   FAILURE: SHAP test failed: {e}")

    # 8. Test Model Saving & SHAP
    print("   Testing Model Saving & SHAP Analysis...")
    try:
        # Mock save input
        model_filename = 'test_xgb_model.json'
        widget.save_model_input.value = model_filename
        
        # Trigger Save
        # Ensure model is trained (Step 4 succeeded)
        if widget.model is not None:
            widget.save_trained_model(None)
            
            # Check file exists
            if os.path.exists(model_filename):
                print(f"   SUCCESS: Model file created at {model_filename}.")
                
                # Test SHAP with file path
                from src.model_analysis import run_shap_analysis
                # Pass filename instead of model object
                run_shap_analysis(model_filename, widget.X_test, widget.class_names)
                print("   SUCCESS: run_shap_analysis executed with file path.")
                
                # Cleanup
                try:
                    os.remove(model_filename)
                except:
                    pass
            else:
                print("   FAILURE: Model file not created.")
        else:
            print("   SKIPPED: Model not trained.")
            
    except Exception as e:
        print(f"   FAILURE: Model Saving/SHAP test failed: {e}")

if __name__ == "__main__":
    test_classification_widget_workflow()
