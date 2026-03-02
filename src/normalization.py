import pandas as pd
import numpy as np
from typing import List, Optional

class Normalizer:
    """
    Standardization and Normalization methodologies.
    Implements MAD (Median Absolute Deviation) and Z-Score scaling.
    """
    
    @staticmethod
    def split_features(df: pd.DataFrame) -> tuple[List[str], List[str]]:
        """Helper to separate Metadata from Feature columns."""
        meta = [c for c in df.columns if c.startswith("Metadata_")]
        feats = [c for c in df.columns if not c.startswith("Metadata_")]
        return meta, feats

    @staticmethod
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-Score Normalization: (X - Mean) / Std
        Applied only to feature columns.
        """
        meta_cols, feat_cols = Normalizer.split_features(df)
        
        if not feat_cols:
            return df
            
        X = df[feat_cols]
        
        # Calculate stats
        mean = X.mean()
        std = X.std()
        
        # Avoid division by zero
        std = std.replace(0, 1)
        
        X_norm = (X - mean) / std
        
        # Reconstruct
        return pd.concat([df[meta_cols], X_norm], axis=1)

    @staticmethod
    def mad_robustize(df: pd.DataFrame, epsilon: float = 1e-18) -> pd.DataFrame:
        """
        MAD Normalization: (X - Median) / MAD
        MAD = median(|Xi - median(X)|)
        """
        meta_cols, feat_cols = Normalizer.split_features(df)
        
        if not feat_cols:
            return df
            
        X = df[feat_cols]
        
        # Calculate Median
        median = X.median()
        
        # Calculate MAD
        # 1.4826 is the scaling factor for normal distribution consistency
        mad = (X - median).abs().median() * 1.4826
        
        # Avoid division by zero
        mad = mad.replace(0, 1) # If MAD is 0, features are constant (or handled by epsilon)
        
        X_norm = (X - median) / (mad + epsilon)
        
        return pd.concat([df[meta_cols], X_norm], axis=1)

def run_normalization(df: pd.DataFrame, method: str = 'mad') -> pd.DataFrame:
    """Wrapper to run specific normalization."""
    if method == 'mad':
        return Normalizer.mad_robustize(df)
    elif method == 'zscore' or method == 'standardize':
        return Normalizer.standardize(df)
    else:
        raise ValueError(f"Unknown method {method}")
