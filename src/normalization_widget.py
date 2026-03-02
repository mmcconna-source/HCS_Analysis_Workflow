import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import numpy as np

class NormalizationWidget:
    """
    Widget for normalizing cell feature data.
    Supports:
    - Normalization to a control (e.g. DMSO)
    - Grouping by Plate (normalize each plate independently)
    """
    
    def __init__(self, input_df=None):
        self.input_df = None
        self.normalized_df = None
        self._create_ui()
        
        if input_df is not None:
            self.load_data(input_df)
        
    def _create_ui(self):
        
        # Settings
        self.dd_plate_col = widgets.Dropdown(description='Plate Col:', options=[], disabled=True)
        self.dd_control_col = widgets.Dropdown(description='Control Col:', options=[], disabled=True)
        self.txt_control_val = widgets.Text(value="DMSO", description='Control Val:', placeholder="e.g. DMSO")
        
        self.dd_method = widgets.Dropdown(
            options=['Percentage of Control (Mean)', 'Percentage of Control (Median)', 'Z-Score (Sample)', 'Z-Score (Control)'],
            value='Percentage of Control (Mean)',
            description='Method:'
        )
        
        self.btn_load_cols = widgets.Button(description="Refresh Columns", button_style='info')
        self.btn_normalize = widgets.Button(description="Run Normalization", button_style='success', layout=widgets.Layout(width='100%'))
        
        # Events
        self.btn_load_cols.on_click(self._refresh_columns)
        self.btn_normalize.on_click(self._run_normalization)
        
        # Layout
        settings_box = widgets.VBox([
            widgets.HTML("<b>Normalization Settings</b>"),
            self.btn_load_cols,
            self.dd_plate_col,
            self.dd_control_col,
            self.txt_control_val,
            widgets.HTML("<hr>"),
            self.dd_method,
            widgets.HTML("<br>"),
            self.btn_normalize
        ], layout=widgets.Layout(padding='10px', border='1px solid #ccc'))
        
        self.out_log = widgets.Output(layout={'height': '200px', 'overflow_y': 'scroll', 'border': '1px solid #eee', 'margin': '10px 0'})
        
        self.main_layout = widgets.VBox([
            widgets.HTML("<h2>Normalization Widget</h2>"),
            settings_box,
            widgets.HTML("<b>Log:</b>"),
            self.out_log
        ])
        
    def display(self):
        display(self.main_layout)
        
    def load_data(self, df):
        """Load dataframe into the widget."""
        # OPTIMIZATION: Downcast float64 to float32 to save memory
        import gc
        self.input_df = df.copy() # Store a copy to avoid mutating external
        
        float_cols = self.input_df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            self.input_df[float_cols] = self.input_df[float_cols].astype('float32')
            with self.out_log: print(f"ℹ️ Optimized memory: Downcasted {len(float_cols)} float64 columns to float32.")
            
        self._refresh_columns(None)
        gc.collect()
        with self.out_log: print(f"✅ Loaded DataFrame with shape {df.shape}")

    def _refresh_columns(self, _):
        if self.input_df is None:
            return
            
        cols = sorted(list(self.input_df.columns))
        
        # Try to guess defaults
        plate_guess = next((c for c in cols if 'plate' in c.lower()), None)
        ctrl_guess = next((c for c in cols if 'cmpd' in c.lower() or 'treatment' in c.lower()), None)
        
        self.dd_plate_col.options = cols
        if plate_guess: self.dd_plate_col.value = plate_guess
        self.dd_plate_col.disabled = False
        
        self.dd_control_col.options = cols
        if ctrl_guess: self.dd_control_col.value = ctrl_guess
        self.dd_control_col.disabled = False
        
    def _run_normalization(self, _):
        import gc
        if self.input_df is None:
            with self.out_log: print("❌ No data loaded.")
            return

        # Clear previous result to free memory BEFORE computing new one
        if self.normalized_df is not None:
             del self.normalized_df
             self.normalized_df = None
             gc.collect()
            
        plate_col = self.dd_plate_col.value
        ctrl_col = self.dd_control_col.value
        ctrl_val = self.txt_control_val.value.strip()
        method = self.dd_method.value
        
        # Identify numeric columns (features) - typically start with 'Cell_' or 'Nuclei_' or imply measurements
        # For simplicity, let's take all numeric types, but maybe filter out Metadata?
        
        numeric_cols = self.input_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude metadata columns and specific technical columns
        excluded_cols = [plate_col, ctrl_col, 'ImageNumber', 'ObjectNumber', 'AreaShape_Center_X', 'AreaShape_Center_Y', 'Location_Center_X', 'Location_Center_Y']
        
        feature_cols = [c for c in numeric_cols if not c.startswith('Metadata_') and c not in excluded_cols]
        
        if not feature_cols:
            with self.out_log: print("❌ No numeric feature columns found (excluding Metadata_).")
            return
            
        with self.out_log: print(f"⏳ Normalizing {len(feature_cols)} features using Group: '{plate_col}' | Method: '{method}'...")
        
        normalized_frames = []
        
        try:
            # Group by Plate
            grouped = self.input_df.groupby(plate_col)
            
            for name, group in grouped:
                # Get Controls
                controls = group[group[ctrl_col] == ctrl_val]
                
                if controls.empty:
                    with self.out_log: print(f"  ⚠️ Plate {name}: No controls found ('{ctrl_val}'). Skipping normalization for this plate.")
                    normalized_frames.append(group) # Keep as is?? Or drop? Usually keep as is or NAN.
                    continue
                
                # Calculate Stats
                # Ensure stats are float32 if input is float32
                if 'Mean' in method:
                    ctrl_stats = controls[feature_cols].mean()
                elif 'Median' in method:
                    ctrl_stats = controls[feature_cols].median()
                
                # Setup Std if needed for Z-Score
                if 'Z-Score' in method:
                    ctrl_mean = controls[feature_cols].mean()
                    ctrl_std = controls[feature_cols].std()
                    # Avoid divide by zero
                    ctrl_std = ctrl_std.replace(0, 1e-9) 
                
                # Apply
                norm_group = group.copy()
                
                if 'Percentage of Control' in method:
                    # Avoid divide by zero if mean is 0
                    ctrl_stats = ctrl_stats.replace(0, np.nan) 
                    norm_group[feature_cols] = (group[feature_cols] / ctrl_stats) * 100
                
                elif 'Z-Score (Control)' in method:
                    norm_group[feature_cols] = (group[feature_cols] - ctrl_mean) / ctrl_std
                    
                # Explicitly downcast result to float32 to keep memory low
                norm_group[feature_cols] = norm_group[feature_cols].astype('float32')
                
                # Store
                normalized_frames.append(norm_group)
                
                # GC per loop might be too aggressive/slow, but for very large plates it helps. 
                # Let's do it only if list gets big? Or just rely on end collection.
            
            # Grouped object can be large, release it
            del grouped
            gc.collect()
            
            self.normalized_df = pd.concat(normalized_frames, ignore_index=True)
            
            # Clear intermediate list
            del normalized_frames
            gc.collect()
            
            with self.out_log: 
                print("✅ Normalization Complete.")
                print(f"   Shape: {self.normalized_df.shape}")
                print("   Sample Data (First 5 cols):")
                print(self.normalized_df[feature_cols[:3]].head())
                
                print("\n👇 RUN THIS IN THE NEXT CELL TO ACCESS DATA 👇")
                print("normalized_df = norm_widget.normalized_df")

        except Exception as e:
            with self.out_log: 
                print(f"❌ Error during normalization: {e}")
                import traceback
                traceback.print_exc()
