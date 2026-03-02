import pandas as pd
from pathlib import Path
from typing import List, Optional, Callable

class DataAggregator:
    """
    Aggregates CellProfiler output CSVs from well-based folder structures.
    """
    
    MERGE_KEYS = ["ImageNumber", "ObjectNumber"]
    
    @staticmethod
    def load_and_prefix(file_path: Path, prefix: str, is_primary: bool = False) -> pd.DataFrame:
        """
        Loads a CSV and prefixes its feature columns.
        Preserves Metadata/Merge keys.
        """
        df = pd.read_csv(file_path, low_memory=False)
        
        # Identify Metadata columns
        metadata_cols = [c for c in df.columns if c.startswith("Metadata_") or c in DataAggregator.MERGE_KEYS]
        
        # If not primary (e.g. Nucleus/Cytoplasm), drop extra metadata to avoid duplicates
        if not is_primary:
            cols_to_drop = [c for c in metadata_cols if c not in DataAggregator.MERGE_KEYS]
            df = df.drop(columns=cols_to_drop, errors="ignore")
            
        # Rename feature columns
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        rename_map = {c: f"{prefix}_{c}" for c in feature_cols}
        df = df.rename(columns=rename_map)
        
        return df

    @staticmethod
    def aggregate_data(root_dir: str, 
                       output_dir: str,
                       cell_csv: str = "MyExpt_Cell.csv",
                       cyto_csv: str = "MyExpt_Cytoplasm.csv",
                       nuc_csv: str = "MyExpt_Nucleus.csv",
                       image_csv: str = "MyExpt_Image.csv") -> pd.DataFrame:
        """
        Walks through the root directory, looks for well folders containing the CSVs,
        and aggregates them.
        """
        root = Path(root_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        all_single_cells = []
        all_images = []
        
        # Assume any subdirectory might be a well folder
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        
        print(f"Scanning {len(subdirs)} folders in {root}...")
        
        for subdir in subdirs:
            well_id = subdir.name
            
            # Paths
            p_cell = subdir / cell_csv
            p_cyto = subdir / cyto_csv
            p_nuc = subdir / nuc_csv
            p_img = subdir / image_csv
            
            # Image Level
            if p_img.exists():
                img_df = pd.read_csv(p_img)
                img_df["Metadata_WellID"] = well_id
                all_images.append(img_df)
                
            # Single Cell Merge
            if p_cell.exists() and p_cyto.exists() and p_nuc.exists():
                try:
                    df_cell = DataAggregator.load_and_prefix(p_cell, "Cell", is_primary=True)
                    df_cyto = DataAggregator.load_and_prefix(p_cyto, "Cytoplasm")
                    df_nuc = DataAggregator.load_and_prefix(p_nuc, "Nucleus")
                    
                    merged = df_cell.merge(df_cyto, on=DataAggregator.MERGE_KEYS).merge(
                        df_nuc, on=DataAggregator.MERGE_KEYS
                    )
                    
                    merged["Metadata_WellID"] = well_id
                    all_single_cells.append(merged)
                except Exception as e:
                    print(f"Error processing well {well_id}: {e}")
        
        master_df = pd.DataFrame()
        if all_single_cells:
            master_df = pd.concat(all_single_cells, ignore_index=True)
            out_file = out_path / "master_single_cell.csv"
            master_df.to_csv(out_file, index=False)
            print(f"Saved Master Single Cell Data ({master_df.shape}) to {out_file}")
            
        if all_images:
            img_master = pd.concat(all_images, ignore_index=True)
            img_master.to_csv(out_path / "master_image_level.csv", index=False)
            
        return master_df
