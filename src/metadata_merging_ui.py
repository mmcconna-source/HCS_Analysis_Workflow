import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import pathlib
import re

class MetadataMergeWidget:
    """
    Widget for merging metadata from a CSV into an existing DataFrame.
    Includes functionality to convert CQ1 naming conventions (W0001) to Standard (A01).
    Supports multiple merge keys (e.g. Metadata_PlateID, Metadata_WellID).
    """
    
    PLATE_CONFIGS = {
        '96-Well': {'rows': 8, 'cols': 12},
        '384-Well': {'rows': 16, 'cols': 24}
    }

    def __init__(self, input_df=None, secondary_df=None):
        """
        Args:
            input_df (pd.DataFrame): The dataframe to merge metadata onto.
            secondary_df (pd.DataFrame): Optional second dataframe to merge metadata from.
        """
        self.input_df = input_df
        self.secondary_df = secondary_df
        self.merged_df = None
        
        self._create_ui()

    def _create_ui(self):
        # --- Section 1: CSV Metadata Merge ---
        self.txt_csv = widgets.Text(placeholder="/path/to/metadata.csv", description='Metadata CSV:', layout=widgets.Layout(width='80%'))
        
        # Split Key Inputs
        self.txt_df_key = widgets.Text(value="Metadata_WellID", description='DF Key(s):', placeholder="Comma separated for multiple", layout=widgets.Layout(width='300px'))
        self.txt_csv_key = widgets.Text(value="WellID", description='CSV Key(s):', placeholder="Comma separated for multiple", layout=widgets.Layout(width='300px'))
        
        self.dd_plate = widgets.Dropdown(options=['96-Well', '384-Well'], value='96-Well', description='Plate Type:', layout=widgets.Layout(width='200px'))
        
        self.chk_convert_df = widgets.Checkbox(value=False, description='Convert Input DF WellIDs (CQ1 -> Std)')
        self.chk_convert_csv = widgets.Checkbox(value=False, description='Convert CSV WellIDs (CQ1 -> Std)')
        
        csv_settings_box = widgets.VBox([
            widgets.HTML("<b>CSV Merge Settings</b>"),
            widgets.HTML("<i>For multiple keys, separate with commas (e.g. 'Metadata_Plate, Metadata_Well'). Order must match.</i>"),
            widgets.VBox([self.txt_df_key, self.txt_csv_key]),
            widgets.HBox([self.dd_plate]),
            self.chk_convert_df,
            self.chk_convert_csv
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='10px 0'))
        
        self.btn_merge_csv = widgets.Button(description="Merge CSV Metadata", button_style='success')
        self.btn_merge_csv.on_click(self._run_merge_csv)
        
        csv_box = widgets.VBox([
            widgets.HTML("<h4>Method 1: Merge from CSV</h4>"),
            self.txt_csv,
            csv_settings_box,
            self.btn_merge_csv
        ], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0'))

        # --- Section 2: Secondary DF Merge ---
        self.txt_sec_key = widgets.Text(value="ImageNumber", description='Merge Key(s):', layout=widgets.Layout(width='300px'))
        
        self.btn_merge_df = widgets.Button(description="Merge Primary -> Secondary", button_style='info')
        self.btn_merge_df.on_click(self._run_merge_df)
        
        df_box = widgets.VBox([
            widgets.HTML("<h4>Method 2: Propagate to Secondary DataFrame</h4>"),
            widgets.HTML("<i>Merges 'Metadata_' columns from the Primary DF (Input/Merged) onto the Secondary DF.</i>"),
            self.txt_sec_key,
            self.btn_merge_df
        ], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0'))
        
        # Common Output
        self.out_log = widgets.Output(layout={'height': '150px', 'overflow_y': 'scroll', 'border': '1px solid #eee'})
        
        self.main_layout = widgets.VBox([
            widgets.HTML("<h3>Metadata Merger Widget</h3>"),
            csv_box,
            df_box,
            widgets.HTML("<b>Log:</b>"),
            self.out_log
        ])

    def display(self):
        display(self.main_layout)

    @staticmethod
    def _cq1_to_standard(well_str, rows, cols):
        """
        Converts CQ1 well ID (e.g. W0001, W1, 1, 001) to Standard (A01).
        Assumes Row-Major ordering (A01, A02... B01...).
        """
        val_str = str(well_str).strip()
        linear_idx = None
        
        # 1. Try pure digit parsing first (Handling "1", "001")
        if val_str.isdigit():
            linear_idx = int(val_str)
        else:
            # 2. Try match with optional W/w prefix (Handling "W0001", "W1", "w01")
            match = re.search(r'^[Ww]?(\d+)$', val_str)
            if match:
                linear_idx = int(match.group(1))
        
        if linear_idx is None:
             return well_str # Not a convertable format
             
        try:
            linear_idx = linear_idx - 1 # 0-indexed
            
            # Row Major Calculation
            r = linear_idx // cols
            c = linear_idx % cols
            
            if r >= rows:
                return well_str # Out of bounds
                
            row_char = chr(65 + r)
            col_str = f"{c+1:02d}"
            
            return f"{row_char}{col_str}"
        except Exception:
            return well_str

    def _parse_keys(self, key_str):
        """Splits comma separated keys into a list."""
        return [k.strip() for k in key_str.split(',') if k.strip()]

    def _run_merge_csv(self, _):
        csv_path = self.txt_csv.value.strip()
        df_keys = self._parse_keys(self.txt_df_key.value)
        csv_keys = self._parse_keys(self.txt_csv_key.value)
        plate_type = self.dd_plate.value
        
        if self.input_df is None:
            with self.out_log: print("❌ No Input Dataframe provided.")
            return

        if not csv_path or not pathlib.Path(csv_path).exists():
             with self.out_log: print(f"❌ Metadata CSV not found: {csv_path}")
             return
             
        if len(df_keys) != len(csv_keys):
            with self.out_log: print(f"❌ Mismatch in number of keys: DF({len(df_keys)}) vs CSV({len(csv_keys)})")
            return

        p_rows = self.PLATE_CONFIGS[plate_type]['rows']
        p_cols = self.PLATE_CONFIGS[plate_type]['cols']
        
        try:
            # 1. Load CSV
            meta_df = pd.read_csv(csv_path)
            
            # Prepare copies to avoid mutating originals/inputs unexpectedly during trial
            df_target = self.input_df.copy()
            df_meta = meta_df.copy()
            
            # 2. Conversions (Only applies to the WellID key if it exists/is selected)
            # Helper to apply conversion
            def apply_conversion(df, col_name, label):
                if col_name not in df.columns:
                     return False
                
                df[col_name] = df[col_name].apply(lambda x: self._cq1_to_standard(x, p_rows, p_cols))
                with self.out_log: print(f"✅ Converted {label} '{col_name}' to Standard.")
                return True

            if self.chk_convert_df.value:
                for key in df_keys:
                    if 'well' in key.lower():
                        if not apply_conversion(df_target, key, "Input DF"):
                             with self.out_log: print(f"⚠️ Could not find '{key}' in Input DataFrame to convert.")

            if self.chk_convert_csv.value:
                for key in csv_keys:
                    if 'well' in key.lower():
                        if not apply_conversion(df_meta, key, "Metadata CSV"):
                             with self.out_log: print(f"⚠️ Could not find '{key}' in Metadata CSV to convert.")
            
            # 3. Check Keys exist
            for k in df_keys:
                if k not in df_target.columns:
                    with self.out_log: print(f"❌ Key '{k}' not found in Input DataFrame.")
                    return
            for k in csv_keys:
                if k not in df_meta.columns:
                    with self.out_log: print(f"❌ Key '{k}' not found in Metadata CSV.")
                    return

            # 4. Standardize CSV Columns
            # Logic: Rename csv_keys -> df_keys (for merging)
            #        Rename ALL other columns -> specific Metadata_ prefix
            
            rename_map = {}
            # Map merge keys first
            for ck, dk in zip(csv_keys, df_keys):
                if ck != dk:
                    rename_map[ck] = dk
            
            # Rename other columns
            for c in df_meta.columns:
                if c in csv_keys:
                    continue # handled above
                
                # Enforce Metadata_ prefix
                if not c.startswith("Metadata_"):
                    # Check if we are renaming it to something that collides?
                    # No, just prefix
                    rename_map[c] = f"Metadata_{c}"
                
            if rename_map:
                df_meta = df_meta.rename(columns=rename_map)

            # 5. Handle Duplicates (to avoid _x, _y)
            # Identify columns in df_meta that are already in df_target AND are not keys
            # We must drop them from df_meta (assuming df_target is source of truth or we just want new columns)
            # Or we merge?
            
            common_cols = set(df_meta.columns) & set(df_target.columns)
            duplicate_cols = [c for c in common_cols if c not in df_keys]
            
            if duplicate_cols:
                with self.out_log: 
                    print(f"⚠️ Duplicate columns found (preserving Input DF values): {duplicate_cols}")
                df_meta = df_meta.drop(columns=duplicate_cols)
                
            # 5. Merge
            self.merged_df = df_target.merge(df_meta, on=df_keys, how='left')
            
            with self.out_log: 
                print(f"✅ CSV Merge Complete!")
                print(f"   Shape: {self.merged_df.shape}")
                print(f"   Keys Used: {df_keys}")
                
                # Check for NaNs in new columns to warn user if merge missed
                new_cols = [c for c in df_meta.columns if c not in df_keys]
                if new_cols:
                    sample_col = new_cols[0]
                    # Check if sample_col is in merged_df (it should be)
                    if sample_col in self.merged_df.columns:
                        missing = self.merged_df[sample_col].isnull().sum()
                        if missing > 0:
                            print(f"   ⚠️ Warning: {missing} rows have NaNs in merged columns (unmatched keys).")

        except Exception as e:
            with self.out_log: print(f"❌ Error during CSV merge: {e}")
            import traceback
            traceback.print_exc()

    def _run_merge_df(self, _):
        """Merge FROM Primary (Input/Merged) TO Secondary."""
        merge_keys = self._parse_keys(self.txt_sec_key.value)
        
        # Source is typically the result of the CSV merge, but fallback to raw input if user skipped that
        source_df = self.merged_df if self.merged_df is not None else self.input_df
        
        if source_df is None:
            with self.out_log: print("❌ No Primary Dataframe available.")
            return

        if self.secondary_df is None:
            with self.out_log: print("❌ No Secondary Dataframe provided.")
            return
            
        try:
            target_df = self.secondary_df.copy()
            # source is already a copy/result, but let's copy to be safe
            src_copy = source_df.copy()
            
            # 1. Validate Keys
            for k in merge_keys:
                if k not in target_df.columns:
                    with self.out_log: print(f"❌ Key '{k}' not found in Secondary DataFrame (Target).")
                    return
                if k not in src_copy.columns:
                    with self.out_log: print(f"❌ Key '{k}' not found in Primary DataFrame (Source).")
                    return
                
            # 2. Select Columns: Keys + Metadata_*
            # "I only want columns that start with Metadata_ to be merged"
            # AND exclude columns that are already in target?? Or overwrite?
            # Typically we propagate metadata, so we want the Metadata columns.
            
            meta_cols = [c for c in src_copy.columns if c.startswith("Metadata_")]
            # Include keys if they aren't metadata prefixed (unlikely if strictly following, but possible)
            keep_cols = list(set(meta_cols + merge_keys))
            
            subset_source = src_copy[keep_cols]
            
            # 3. Merge Source -> Target
            # We duplicate columns if they exist in target (except keys).
            # Let's drop them from source if they exist in target? 
            # Or assume source is "truth" for metadata?
            # Standard merge with suffixes will happen.
            
            self.merged_df = target_df.merge(subset_source, on=merge_keys, how='left')
            
            with self.out_log:
                print(f"✅ Propagated Metadata to Secondary DF!")
                print(f"   Shape: {self.merged_df.shape}")
                print(f"   Keys: {merge_keys}")
                added = [c for c in self.merged_df.columns if c not in target_df.columns]
                print(f"   Columns Added: {added[:5]} ...")

        except Exception as e:
             with self.out_log: print(f"❌ Error during DF merge: {e}")
             import traceback
             traceback.print_exc()
