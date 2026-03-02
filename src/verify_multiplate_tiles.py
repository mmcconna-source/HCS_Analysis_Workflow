import sys
import unittest
import pandas as pd
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

# Add src to path
sys.path.append('src')

from tile_extraction import NotebookConfig, resolve_image_paths, ConfigurationError

class TestMultiPlatePaths(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for plates
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        
        self.plate1_dir = self.root / "Plate1"
        self.plate2_dir = self.root / "Plate2"
        self.plate1_dir.mkdir()
        self.plate2_dir.mkdir()
        
        # Create dummy image files
        # Pattern: {well}_F{field:04d}_T0001_Z0001_C{channel:02d}.tif
        # Plate 1: Well A01, Field 1
        (self.plate1_dir / "A01_F0001_T0001_Z0001_C01.tif").touch()
        (self.plate1_dir / "A01_F0001_T0001_Z0001_C02.tif").touch()
        
        # Plate 2: Well B02, Field 1
        (self.plate2_dir / "B02_F0001_T0001_Z0001_C01.tif").touch()
        
        # Setup Dataframe
        data = {
            'Metadata_PlateID': ['Plate1', 'Plate2'],
            'Metadata_WellID': ['A01', 'B02'],
            'Metadata_Field': [1, 1]
        }
        self.df = pd.DataFrame(data)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_dict_config_resolution(self):
        print("Testing Multi-Plate Dictionary Resolution...")
        
        # Config with Dict
        config = NotebookConfig(
            csv_path="dummy.csv", # Not used for path resolution
            image_base_path={
                'Plate1': str(self.plate1_dir),
                'Plate2': str(self.plate2_dir)
            },
            channel_names=['C1', 'C2'],
            filename_pattern="{well}_F{field:04d}_T0001_Z0001_C{channel:02d}.tif"
        )
        
        # Test Plate 1 Row
        row1 = self.df.iloc[0]
        paths1 = resolve_image_paths(row1, config)
        
        expected_p1 = self.plate1_dir / "A01_F0001_T0001_Z0001_C01.tif"
        self.assertEqual(paths1['C1'], expected_p1)
        print("✅ Plate 1 resolved correctly.")
        
        # Test Plate 2 Row
        row2 = self.df.iloc[1]
        paths2 = resolve_image_paths(row2, config)
        
        expected_p2 = self.plate2_dir / "B02_F0001_T0001_Z0001_C01.tif"
        self.assertEqual(paths2['C1'], expected_p2)
        print("✅ Plate 2 resolved correctly.")
        
    def test_missing_plate_key(self):
        print("Testing Missing Plate Key Error...")
        
        # Config missing Plate2
        config = NotebookConfig(
            csv_path="dummy.csv",
            image_base_path={
                'Plate1': str(self.plate1_dir)
                # Plate2 missing
            },
            channel_names=['C1'],
            filename_pattern="{well}_F{field:04d}_T0001_Z0001_C{channel:02d}.tif"
        )
        
        row2 = self.df.iloc[1] # Plate2
        
        with self.assertRaises(ConfigurationError):
            resolve_image_paths(row2, config)
        print("✅ Correctly raised error for missing plate key.")
            
    def test_legacy_string_config(self):
        print("Testing Legacy String Config...")
        
        config = NotebookConfig(
            csv_path="dummy.csv",
            image_base_path=str(self.plate1_dir), # Single string
            channel_names=['C1'],
            filename_pattern="{well}_F{field:04d}_T0001_Z0001_C{channel:02d}.tif"
        )
        
        row1 = self.df.iloc[0]
        paths = resolve_image_paths(row1, config)
        self.assertEqual(paths['C1'], self.plate1_dir / "A01_F0001_T0001_Z0001_C01.tif")
        print("✅ Legacy string config still works.")

if __name__ == '__main__':
    unittest.main()
