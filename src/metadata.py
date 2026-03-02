import pandas as pd
from pathlib import Path
from typing import Union, List

class MetadataMerger:
    """
    Handles merging of experimental metadata with feature data.
    """
    
    @staticmethod
    def merge_metadata(feature_df: pd.DataFrame, 
                       metadata: Union[str, Path, pd.DataFrame], 
                       join_on: str = 'Metadata_WellID') -> pd.DataFrame:
        """
        Merges metadata onto the feature DataFrame.
        
        Args:
            feature_df: The DataFrame containing cellular features.
            metadata: Path to metadata CSV or DataFrame.
            join_on: Column to join on (usually 'Metadata_WellID').
        """
        if isinstance(metadata, (str, Path)):
            if not Path(metadata).exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata}")
            meta_df = pd.read_csv(metadata)
        else:
            meta_df = metadata.copy()
            
        # Ensure join keys match type (String is safest for IDs like "01")
        if join_on not in feature_df.columns:
            raise ValueError(f"Join key {join_on} not found in Feature Data.")
            
        # Add 'Metadata_' prefix if missing to metadata columns
        # (Exclude the join key from renaming if it matches exactly, or handle carefully)
        # Strategy: Rename all, then ensure join key aligns
        
        # 1. safe join key prep
        feature_df[join_on] = feature_df[join_on].astype(str)
        
        # 2. Prepare metadata
        # Detect if 'WellID' exists but not 'Metadata_WellID'
        if join_on not in meta_df.columns and 'WellID' in meta_df.columns:
             meta_df = meta_df.rename(columns={'WellID': join_on})

        if join_on not in meta_df.columns:
            raise ValueError(f"Join key {join_on} not found in Metadata.")
            
        meta_df[join_on] = meta_df[join_on].astype(str)
        
        # 3. Add Prefix to other columns
        new_cols = {}
        for c in meta_df.columns:
            if c != join_on and not c.startswith("Metadata_"):
                new_cols[c] = f"Metadata_{c}"
        meta_df = meta_df.rename(columns=new_cols)
        
        # 4. Merge
        merged = feature_df.merge(meta_df, on=join_on, how='left')
        
        # 5. Check
        if merged[join_on].isnull().any():
            print(f"Warning: Some rows failed to find matching metadata for {join_on}")
            
        return merged
