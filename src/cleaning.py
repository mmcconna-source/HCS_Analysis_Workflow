import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from typing import List, Optional

class DataCleaner:
    """
    Cleans data by removing excluded columns, handling NaNs/Infs, 
    and filtering low variance features.
    """
    
    DEFAULT_EXCLUSIONS = [
        'ImageNumber', 'ObjectNumber', 'Parent', 'Children', 
        'Location', 'PathName', 'FileName', 'Digest', 'URL', 'ExecutionTime',
        'Center_X', 'Center_Y', 'BoundingBox', 'EulerNumber'
    ]
    
    @staticmethod
    def clean_data(df: pd.DataFrame, 
                   variance_threshold: float = 0.0,
                   custom_exclusions: Optional[List[str]] = None,
                   impute_nans: bool = True) -> pd.DataFrame:
        """
        Performs standard cleaning pipeline.
        """
        data = df.copy()
        print(f"Initial Shape: {data.shape}")
        
        # 1. Drop Empty Columns
        data = data.dropna(axis=1, how='all')
        
        # 2. Exclude unwanted columns
        exclusions = DataCleaner.DEFAULT_EXCLUSIONS + (custom_exclusions or [])
        
        # Identify columns to drop (substring match or exact?)
        # User script used substring matching. We will be safer with exact + specific substrings.
        # User Logic: "if 'BoundingBox' in col..."
        
        cols_to_drop = []
        for col in data.columns:
            # Keep Metadata
            if col.startswith("Metadata_"):
                continue
                
            # Check exclusions
            is_excluded = False
            for exc in exclusions:
                if exc in col:
                    is_excluded = True
                    break
            
            if is_excluded:
                cols_to_drop.append(col)
                
        data = data.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} excluded columns.")
        
        # 3. Numeric Selection
        # Separate Metadata/Index from Features
        meta_cols = [c for c in data.columns if c.startswith("Metadata_") or c.startswith("Image_")] # Keep Image_? usually stats
        # Re-evaluate Feature columns 
        # (Assuming 'Image_' might be feature or meta, user script excludes 'Image_Intensity' in exclusion list)
        
        numeric_df = data.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric_df.columns if c not in meta_cols]
        
        if not feature_cols:
             raise ValueError("No numeric feature columns remaining.")

        X = data[feature_cols].copy()
        
        # 4. Handle Inf/NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if impute_nans:
            X = X.fillna(0) # Simple imputation as per user script Option B
        else:
            X = X.dropna()
            # Align data (complex if rows dropped)
            data = data.loc[X.index]
            
        # 5. Variance Threshold
        if variance_threshold >= 0:
            selector = VarianceThreshold(threshold=variance_threshold)
            try:
                X_clean = selector.fit_transform(X)
                # Recover names
                mask = selector.get_support()
                selected_feats = np.array(feature_cols)[mask]
                
                # Update X
                X = pd.DataFrame(X_clean, columns=selected_feats, index=X.index)
                print(f"Variance Threshold removed {len(feature_cols) - len(selected_feats)} features.")
            except ValueError:
                 print("Variance selection failed (possibly empty result). Keeping all.")
        
        # 6. Reconstruct
        # Keep non-feature columns + new feature columns
        non_feature = data.drop(columns=feature_cols)
        final_df = pd.concat([non_feature.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        
        print(f"Final Shape: {final_df.shape}")
        return final_df
