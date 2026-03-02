
import sys
import unittest.mock as mock
from pathlib import Path
import pandas as pd
import numpy as np
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent))

from umap_exploration_widget import UMAPExplorationWidget
from tile_extraction import NotebookConfig

def test_random_sampling():
    print("\nTesting Random Sampling Logic...")
    
    # 1. Setup Data
    df = pd.DataFrame({'A': range(100), 'UMAP1': range(100), 'UMAP2': range(100)})
    config = NotebookConfig(csv_path='dummy', image_base_path='dummy')
    
    widget = UMAPExplorationWidget(df, config, output_root="TEST_EXPORT")
    
    # Select all
    widget.selected_indices = list(range(100))
    
    # Configure Random Sample
    widget.random_sample_checkbox.value = True
    widget.max_tiles_input.value = 10
    
    # Mock ChannelMappingWidget to avoid UI blocking
    with mock.patch('umap_exploration_widget.ChannelMappingWidget') as MockCM:
        MockCM.return_value.widget_container = mock.MagicMock()
        
        # Run Generation (Populates _export_indices)
        # We also need to mock extract_multichannel_tile to avoid image errors during preview generation
        with mock.patch('umap_exploration_widget.extract_multichannel_tile', return_value=np.zeros((10,10,3))):
            widget.generate_tiles_for_selection(None)
            
            # Check results
            if hasattr(widget, '_export_indices'):
                n_exported = len(widget._export_indices)
                print(f"   Export indices count: {n_exported}")
                if n_exported == 10:
                    print("   SUCCESS: Correctly sampled 10 items.")
                else:
                    print(f"   FAILURE: Expected 10 items, got {n_exported}")
            else:
                print("   FAILURE: _export_indices not set.")

def test_pptx_export_call():
    print("\nTesting PPTX Export Logic (Mocked)...")
    
    df = pd.DataFrame({
        'Well': ['A01'], 'Field': [1], 'ObjectNumber': [1],
        'UMAP1': [0], 'UMAP2': [0]
    })
    config = NotebookConfig(csv_path='dummy', image_base_path='dummy')
    widget = UMAPExplorationWidget(df, config, output_root="TEST_EXPORT")
    
    # Setup state
    widget._export_indices = [0]
    widget.export_format_dropdown.value = 'PowerPoint (PPTX)'
    widget.save_name_input.value = "test_run"
    
    # Mock mocks
    with mock.patch('umap_exploration_widget.ChannelMappingWidget') as MockCM:
         MockCM.return_value.get_mappings.return_value = {}
         widget.cm_widget = MockCM.return_value
         
         # Mock internal _export_to_pptx to verify it gets called
         with mock.patch.object(widget, '_export_to_pptx') as mock_export:
             widget._on_confirm_export(None)
             
             if mock_export.called:
                 print("   SUCCESS: _export_to_pptx was called when PPTX format selected.")
             else:
                 print("   FAILURE: _export_to_pptx was NOT called.")

def test_pptx_export_implementation():
    print("\nTesting _export_to_pptx Implementation (Deep Mock)...")
    
    df = pd.DataFrame({
        'Well': ['A01'], 'Field': [1], 'ObjectNumber': [1],
        'UMAP1': [0], 'UMAP2': [0]
    })
    config = NotebookConfig(
        csv_path='dummy', image_base_path='dummy',
        well_column='Well', field_column='Field'
    )
    widget = UMAPExplorationWidget(df, config, output_root="TEST_EXPORT")
    
    # Mock sys.modules for pptx
    sys.modules['pptx'] = mock.MagicMock()
    sys.modules['pptx.util'] = mock.MagicMock()
    
    # Mock extract and rgb creation
    with mock.patch('umap_exploration_widget.extract_multichannel_tile', return_value=np.zeros((10,10,3))), \
         mock.patch('umap_exploration_widget.create_rgb_composite', return_value=np.zeros((10,10,3), dtype=np.uint8)):
         
         try:
             widget._export_to_pptx(df, config, {}, "test.pptx")
             print("   SUCCESS: _export_to_pptx ran without error (with mocked pptx).")
         except Exception as e:
             print(f"   FAILURE: _export_to_pptx crashed: {e}")
             import traceback
             traceback.print_exc()

if __name__ == "__main__":
    if Path("TEST_EXPORT").exists():
        shutil.rmtree("TEST_EXPORT")
        
    test_random_sampling()
    test_pptx_export_call()
    test_pptx_export_implementation()
