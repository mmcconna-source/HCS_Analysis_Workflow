import sys
import pandas as pd
import numpy as np
import gc
import unittest
from unittest.mock import MagicMock

# Add src to path
sys.path.append('src')

from data_filtering_widget import DataFilteringWidget

class TestDataFilteringOptimization(unittest.TestCase):
    def setUp(self):
        # Create a large DataFrame to test memory handling
        # 2000 rows, 500 columns (approx 8MB, small enough for quick test but structured for optimization logic)
        # We need float64 to test downcasting
        print("Generating dummy data...")
        self.rows = 2000
        self.cols = 500
        self.data = np.random.randn(self.rows, self.cols)
        self.df = pd.DataFrame(self.data, columns=[f'Cell_Feature_{i}' for i in range(self.cols)])
        
        # Add some metadata and technical columns
        self.df['Metadata_Plate'] = 'Plate1'
        self.df['ImageNumber'] = range(self.rows)
        self.df['ObjectNumber'] = range(self.rows)
        
        # Mock IPython get_ipython().user_global_ns
        self.mock_ip = MagicMock()
        self.mock_ip.user_global_ns = {'test_df': self.df}
        
    def test_optimization_logic(self):
        print("Testing Optimization Logic...")
        
        widget = DataFilteringWidget()
        
        # Mocking the dataframe selection
        widget.df_selector.options = ['test_df']
        widget.df_selector.value = 'test_df'
        widget.output_name_text.value = 'filtered_df_output'
        
        # Mock IPython import within the method
        import sys
        module_name = 'IPython'
        if module_name not in sys.modules:
            sys.modules[module_name] = MagicMock()
            sys.modules[module_name].get_ipython = MagicMock(return_value=self.mock_ip)
        else:
            # If it exists (e.g. valid environment), we might need to patch it
            # But for this standalone script, the simple import mock might fail if IPython isn't installed in the output env
            # Let's rely on the method using the real IPython if available? 
            # Or better, monkeypatch get_ipython in the module if possible, or just the mock in sys.modules
            sys.modules[module_name].get_ipython = MagicMock(return_value=self.mock_ip)

        # Run Apply
        # We capture stdout to check for optimization messages
        from io import StringIO
        import sys
        
        # Method calls print(), so we just run it and assume no crash
        try:
             widget.on_apply_click(None)
             
             # Check if output exists in global ns
             self.assertIn('filtered_df_output', self.mock_ip.user_global_ns)
             result_df = self.mock_ip.user_global_ns['filtered_df_output']
             
             # Verify Float32 Downcasting
             # Check the first feature column
             dtype = result_df['Cell_Feature_0'].dtype
             print(f"Resulting Dtype: {dtype}")
             self.assertTrue(dtype == 'float32', "Numeric columns should be downcasted to float32")
             
             # Verify Technical Columns Dropped
             self.assertNotIn('ImageNumber', result_df.columns, "ImageNumber should be dropped")
             
             # Verify Metadata Kept
             self.assertIn('Metadata_Plate', result_df.columns, "Metadata_Plate should be kept")
             
             print("✅ Optimization Verification Passed")
             
        except Exception as e:
            self.fail(f"Optimization test failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
