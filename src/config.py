import os
from pathlib import Path

class Config:
    """
    Central configuration for the Single Cell Analysis Pipeline.
    All paths and constants should be defined here.
    """
    # Base Directories (Update these paths as needed or load from env variables)
    BASE_DIR = Path(r"z:\Active_Users_Data\Matthew\Analysis_Pipeline_Tool")
    DATA_DIR = BASE_DIR / "Example_Data"
    OUTPUT_DIR = BASE_DIR / "Output"
    
    # Module 1: Preprocessing
    # -----------------------
    # Regex for Yokogawa Image Naming: Match W (Well), F (Field), C (Channel)
    FILENAME_REGEX = r"(W\d+)(F\d+).*?(C\d+)"
    
    # Channel Map: Map C numbers to Color/Name
    # Example: {'C1': 'Blue', 'C2': 'Green', 'C3': 'Red'}
    CHANNEL_MAP = {
        'C1': 'Blue',
        'C2': 'Green', 
        'C3': 'Red'
    }
    
    # Variance Threshold for Feature Selection
    VARIANCE_THRESHOLD = 0.0
    
    # Metadata Column Prefix
    METADATA_PREFIX = "Metadata_"
    
    @staticmethod
    def ensure_dirs():
        """Ensure critical directories exist."""
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    Config.ensure_dirs()
    print(f"Configuration loaded. Base Dir: {Config.BASE_DIR}")
