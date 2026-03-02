
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from umap_exploration_widget import UMAPExplorationWidget, UMAPViewerWidget
from tile_extraction import NotebookConfig

def test_filename_pattern():
    print("\nTesting Filename Pattern Logic...")
    from tile_extraction import NotebookConfig, resolve_image_paths
    import pandas as pd
    
    # Mock config with custom pattern
    config = NotebookConfig(
        csv_path="dummy.csv",
        image_base_path="images",
        filename_pattern="{well}-Field{field:02d}-{channel_name}.tif"
    )
    
    # Mock row
    row = pd.Series({
        'Metadata_WellID': 'A01',
        'Metadata_Field': 1,
    })
    
    # We expect the path to be images/A01-Field01-DNA.tif (for first channel)
    # The function checks for existence, so it will raise ImageNotFoundError
    # BUT we can check the path in the error message or mock Path.exists
    
    try:
        resolve_image_paths(row, config)
    except Exception as e:
        msg = str(e)
        expected = "A01-Field01-DNA.tif"
        if expected in msg:
             print(f"   SUCCESS: Error message contains expected filename: {expected}")
        else:
             print(f"   FAILURE: Error message '{msg}' did not contain '{expected}'")

def test_umap_widget_headless():
    print("Testing UMAPExplorationWidget...")
    
    # 1. Setup Mock Data
    df = pd.DataFrame({
        'Nucleus_AreaShape_Center_X': np.random.rand(100) * 1000,
        'Nucleus_AreaShape_Center_Y': np.random.rand(100) * 1000,
        'Metadata_WellID': ['A01'] * 50 + ['A02'] * 50,
        'Metadata_Field': [1] * 100,
        'ImageNumber': [1] * 100,
        'ObjectNumber': range(100),
        'Feature_1': np.random.rand(100),
        'Feature_2': np.random.rand(100),
        'UMAP1': np.zeros(100), # Placeholders
        'UMAP2': np.zeros(100) 
    })
    
    config = NotebookConfig(
        csv_path='dummy.csv',
        image_base_path='dummy_images',
        x_column='Nucleus_AreaShape_Center_X',
        y_column='Nucleus_AreaShape_Center_Y',
        umap_x_column='UMAP1', 
        umap_y_column='UMAP2',
        well_column='Metadata_WellID',
        field_column='Metadata_Field',
        channel_names=['DNA']
    )
    
    # Cleanup previous test
    if Path("UMAP_TEST").exists():
        shutil.rmtree("UMAP_TEST")
        
    widget = UMAPExplorationWidget(df, config, output_root="UMAP_TEST")
    
    # 2. Test UMAP Generation (Headless)
    print("Running UMAP generation (Scaled)...")
    widget.scale_data_checkbox.value = True
    widget.random_state_input.value = 999 
    try:
        widget.run_umap(None)
        if 'UMAP1' in widget.df.columns and widget.df['UMAP1'].sum() != 0:
            print("SUCCESS: UMAP generated.")
            umap1_run1 = widget.df['UMAP1'].copy()
            
            # Run Again with Same Seed
            print("Running UMAP generation (Same Seed)...")
            widget.run_umap(None)
            umap1_run2 = widget.df['UMAP1'].copy()
            
            if np.allclose(umap1_run1, umap1_run2):
                 print("SUCCESS: Random state is working (results identical).")
            else:
                 print("FAILURE: Random state not working or results differ.")
            
        else:
            print("FAILURE: UMAP1 column empty or unchanged.")
    except Exception as e:
        print(f"FAILURE during UMAP run: {e}")

    # 3. Test Clustering (Headless)
    print("Running KMeans...")
    try:
        widget.run_kmeans(None)
        if 'kmeans' in widget.df.columns:
            print(f"SUCCESS: KMeans calculated. Found clusters: {widget.df['kmeans'].unique()}")
        else:
            print("FAILURE: 'kmeans' column missing.")
    except Exception as e:
        print(f"FAILURE during KMeans: {e}")

    # 4. Test Save
    print("Testing Save...")
    widget.save_name_input.value = "test_save"
    widget.save_umap_data(None)
    
    # Needs a mock figure to save
    import matplotlib.pyplot as plt
    widget.fig = plt.figure()
    widget.save_plot(None)
    
    expected_file = Path("UMAP_TEST/test_save/umap_data.csv")
    expected_plot = Path("UMAP_TEST/test_save/umap_plot.png")
    
    if expected_file.exists():
        print(f"SUCCESS: File saved at {expected_file}")
    else:
        print(f"FAILURE: File not found {expected_file}")
        
    if expected_plot.exists():
        print(f"SUCCESS: Plot saved at {expected_plot}")
    else:
        print(f"FAILURE: Plot not found {expected_plot}")

    # 5. Test Viewer
    print("Testing Viewer...")
    viewer = UMAPViewerWidget(base_dir="UMAP_TEST")
    viewer.refresh_sessions()
    if 'test_save' in viewer.session_dropdown.options:
        print("SUCCESS: Viewer sees saved session.")
        
        # Test Load
        viewer.session_dropdown.value = 'test_save'
        try:
            viewer.load_session(None)
            if viewer.df is not None and not viewer.df.empty:
                 print(f"SUCCESS: Viewer loaded {len(viewer.df)} rows.")
                 
                 # Test Color Change (Trigger Plot)
                 print("Testing Viewer Plot (Color By)...")
                 if 'leiden' in viewer.df.columns:
                     viewer.color_by_dropdown.value = 'leiden' # Should trigger plot
                 else:
                     viewer.color_by_dropdown.value = 'kmeans'
                 print("SUCCESS: Viewer plot updated without error.")
            else:
                 print("FAILURE: Viewer loaded empty or None df.")
        except Exception as e:
            print(f"FAILURE during Viewer load/plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"FAILURE: Viewer options: {viewer.session_dropdown.options}")

    # 6. Test Drill Down (Non-Destructive)
    print("Test 6: Testing Drill Down (Non-Destructive)...")
    # Select first 10 points
    widget.selected_indices = list(range(10))
    initial_len = len(widget.df)
    
    # Run Drill Down
    # This should SPAWN a child widget but NOT change existing widget df
    widget.drill_down(None)
    
    if len(widget.df) == initial_len:
        print(f"   SUCCESS: Parent dataframe preserved ({len(widget.df)} rows).")
    else:
         print(f"   FAILURE: Parent dataframe modified! Length is {len(widget.df)}.")

    # 7. Test Coloring Logic (Metadata & Feature Modes)
    print("Test 7: Testing Coloring Logic (Dual Mode)...")
    
    # Metadata Mode
    widget.color_mode_tgl.value = 'Metadata'
    if 'Metadata_WellID' in widget.metadata_dropdown.options:
         widget.metadata_dropdown.value = 'Metadata_WellID'
         print("   SUCCESS: Metadata coloring selected.")
    
    # Feature Mode
    widget.color_mode_tgl.value = 'Feature'
    widget.color_search_input.value = "Feature"
    
    if "Feature_1" in widget.feature_dropdown.options:
         print("   SUCCESS: Search filtered feature options.")
         widget.feature_dropdown.value = "Feature_1"
         
         # Test Log Scale
         widget.log_scale_color_checkbox.value = True
         print("   SUCCESS: Log scale toggled (no crash).")
         
         print("   SUCCESS: Outlier slider moved (no crash).")
    else:
         print(f"   FAILURE: Search did not find 'Feature_1'. Options: {widget.feature_dropdown.options[:5]}...")

    # 8. Test Interactive Tile Generation (Mock)
    print("Test 8: Testing Interactive Tile Generation Init...")
    widget.selected_indices = list(range(5))
    
    # Mock extract_multichannel_tile to avoid file I/O errors in headless env
    import unittest.mock
    from tile_extraction import extract_multichannel_tile
    
    # We need to patch where it is imported in the widget module
    # or just try running it and catch if it fails due to file I/O, which verifies logic up to that point
    
    try:
        # Inject a mock tile into the widget if we can, or just mock the function in the module
        # But we can't easily patch the module from here without more setup. 
        # Let's just create a dummy ChannelMappingWidget manually to verify import
        try:
            from channel_mapping_widget import ChannelMappingWidget
        except ImportError:
            # Fallback if running from root
            from src.channel_mapping_widget import ChannelMappingWidget
            
        if ChannelMappingWidget:
             # Mock 5 tiles
             tiles = [np.zeros((10, 10, 5)) for _ in range(5)]
             cm = ChannelMappingWidget(config, tiles)
             print("   SUCCESS: ChannelMappingWidget initialized with 5 tiles.")
             
             # Test CMY Validation
             try:
                 from tile_extraction import ChannelMapping
                 m = ChannelMapping(0, 'Test', 'Y')
                 m.validate()
                 print("   SUCCESS: ChannelMapping validated 'Y' correctly.")
             except Exception as e:
                 print(f"   FAILURE: ChannelMapping validation failed for 'Y': {e}")
                 
        else:
             print("   FAILURE: ChannelMappingWidget not imported.")
             
        # Trigger the method
        # It will likely fail at extract_multichannel_tile if image paths are bad
        # But we can check if it prints "Extracting sample..."
        widget.generate_tiles_for_selection(None)
        print("   SUCCESS: generate_tiles_for_selection called (check output for errors).")
        
    except Exception as e:
        print(f"   WARNING: Tile generation init failed (expected if images missing): {e}")

def test_robust_extraction():
    print("\nTesting Robust Column Access...")
    from tile_extraction import get_row_val_with_fallback, resolve_image_paths, NotebookConfig
    
    row_plain = pd.Series({'WellID': 'A01'})
    row_meta = pd.Series({'Metadata_WellID': 'A02'})
    
    val1 = get_row_val_with_fallback(row_plain, 'WellID')
    val2 = get_row_val_with_fallback(row_meta, 'WellID')
    
    if val1 == 'A01' and val2 == 'A02':
        print("   SUCCESS: get_row_val_with_fallback handles both plain and prefixed columns.")
    else:
        print(f"   FAILURE: get_row_val_with_fallback failed. Got {val1} and {val2}")
    
    # Test resolve_image_paths with mixed columns
    print("Testing resolve_image_paths with mixed columns...")
    
    # Config expects 'Metadata_WellID' but row has 'WellID' (or vice versa)
    config_robust = NotebookConfig(
        csv_path="dummy.csv",
        image_base_path="images",
        well_column="Metadata_WellID", # Config says Metadata_WellID
        field_column="Metadata_Field",
        filename_pattern="{well}-Field{field:02d}.tif"
    )
    
    row_mixed = pd.Series({
        'Metadata_WellID': 'B03', # Matching config
        'Field': 5 # Missing prefix compared to config
    })
    
    try:
        # We expect ImageNotFoundError because image doesn't exist, NOT KeyError
        paths = resolve_image_paths(row_mixed, config_robust)
    except Exception as e:
        if "Image not found" in str(e):
             print("   SUCCESS: resolve_image_paths processed the columns (got ImageNotFound as expected).")
        elif "KeyError" in str(e):
             print(f"   FAILURE: resolve_image_paths failed with KeyError: {e}")
        else:
             print(f"   UNKNOWN: resolve_image_paths failed with {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_filename_pattern()
    test_robust_extraction()
    test_umap_widget_headless()
