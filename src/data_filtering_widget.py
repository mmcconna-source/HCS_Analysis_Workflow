import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import re
from sklearn.feature_selection import VarianceThreshold

class DataFilteringWidget:
    def __init__(self):
        self._df = None
        
        # --- UI Components ---
        
        # Dataframe Selector
        self.df_selector = widgets.Dropdown(
            options=[],
            description='Select DF:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        self.refresh_btn = widgets.Button(
            description='Refresh',
            icon='refresh',
            layout=widgets.Layout(width='100px')
        )
        self.refresh_btn.on_click(self.refresh_dataframe_list)
        
        # Output Name
        self.output_name_text = widgets.Text(
            value='filtered_df',
            description='Output Name:',
            placeholder='Enter variable name without quotes',
            style={'description_width': 'initial'}
        )
        
        # Filters
        self.drop_empty_cols = widgets.Checkbox(
            value=True,
            description='Drop Empty/NA Columns',
            indent=False
        )
        
        self.blocklist_features = widgets.Checkbox(
            value=True,
            description='Blocklist Features (Metadata, Location, etc.)',
            indent=False
        )

        self.custom_drop_cols = widgets.Textarea(
            value='',
            placeholder='Col1, Col2, Col3',
            description='Drop Specific Cols:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='60px', width='95%')
        )

        self.custom_drop_patterns = widgets.Textarea(
            value='',
            placeholder='Pattern1, Pattern2',
            description='Drop Patterns:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='60px', width='95%')
        )
        
        self.handle_nans = widgets.Dropdown(
            options=['None', 'Drop Rows', 'Fill with 0'],
            value='Drop Rows',
            description='Handle NaNs/Infs:',
            style={'description_width': 'initial'}
        )
        
        self.variance_threshold_toggle = widgets.Checkbox(
            value=True,
            description='Remove Low Variance',
            indent=False
        )
        self.variance_threshold_value = widgets.FloatText(
            value=0.0,
            description='Threshold:',
            layout=widgets.Layout(width='150px')
        )
        
        self.correlation_threshold_toggle = widgets.Checkbox(
            value=True,
            description='Remove High Correlation',
            indent=False
        )
        self.correlation_threshold_value = widgets.FloatText(
            value=0.95,
            description='Threshold:',
            layout=widgets.Layout(width='150px')
        )
        
        self.apply_btn = widgets.Button(
            description='Apply Filters',
            button_style='success',
            icon='check'
        )
        self.apply_btn.on_click(self.on_apply_click)
        
        self.output_area = widgets.Output()
        
        # Layout
        self.widget = widgets.VBox([
            widgets.HBox([self.df_selector, self.refresh_btn]),
            self.output_name_text,
            widgets.HTML("<b>Filtering Options:</b>"),
            self.drop_empty_cols,
            self.blocklist_features,
            self.custom_drop_cols,
            self.custom_drop_patterns,
            self.handle_nans,
            self.handle_nans,
            widgets.HBox([self.variance_threshold_toggle, self.variance_threshold_value]),
            widgets.HBox([self.correlation_threshold_toggle, self.correlation_threshold_value]),
            widgets.HTML("<br>"),
            self.apply_btn,
            self.output_area
        ])
        
        self.refresh_dataframe_list()

    def refresh_dataframe_list(self, b=None):
        # This function effectively only works when run in a notebook where globals() helps
        # But directly accessing globals() inside a class in a utility script is tricky
        # The user will likely have to pass the globals dictionary or we rely on the user running this.
        # Alternatively, we can inspect common global variables if we can access the memory, but that's hard.
        # Strategy: Ask user to assign self.dfs = globals() before using, or pass it in init
        # For now, we'll try to get it from the ipython shell if available
        try:
             from IPython import get_ipython
             ip = get_ipython()
             if ip:
                gl = ip.user_global_ns
                dfs = [var for var, val in gl.items() if isinstance(val, pd.DataFrame) and not var.startswith('_')]
                self.df_selector.options = dfs
        except Exception:
            self.df_selector.options = ['No IPython Env']

    def display(self):
        display(self.widget)
        
    def on_apply_click(self, b):
        with self.output_area:
            clear_output()
            df_name = self.df_selector.value
            output_name = self.output_name_text.value
            
            if not df_name or not output_name:
                print("Error: Please select a dataframe and specify an output name.")
                return
            
            # Retrieve dataframe from globals
            try:
                from IPython import get_ipython
                import gc
                ip = get_ipython()
                # OPTIMIZATION: Do not create a full copy immediately if we can avoid it, 
                # but we need to modify it. So we must copy.
                # However, we can immediately downcast to float32 to save space.
                original_df = ip.user_global_ns[df_name]
            except Exception as e:
                print(f"Error accessing dataframe: {e}")
                return
            
            print(f"Starting filtering on {df_name}...")
            print(f"Initial shape: {original_df.shape}")
            
            # 1. OPTIMIZATION: Create copy and Downcast immediately
            # Use float32 for numeric columns to save 50% RAM
            final_df = original_df.copy()
            
            # Identify float64 columns and downcast
            float_cols = final_df.select_dtypes(include=['float64']).columns
            if len(float_cols) > 0:
                final_df[float_cols] = final_df[float_cols].astype('float32')
                print(f"- Optimized memory: Downcasted {len(float_cols)} float64 columns to float32.")
                
            gc.collect()
            
            # 2. Drop Empty Columns
            if self.drop_empty_cols.value:
                initial_cols = final_df.shape[1]
                final_df = final_df.dropna(axis=1, how='all')
                print(f"- Dropped {initial_cols - final_df.shape[1]} empty columns.")
                
            # 3. Blocklist Features & Technical Columns
            if self.blocklist_features.value:
                # Default extensive blocklist
                blocklist = ['ImageNumber', 'ObjectNumber', 'Parent', 'Children', 'Location', 
                             'PathName', 'FileName', 'Digest', 'URL', 'ExecutionTime', 
                             'Center_X', 'Center_Y', 'BoundingBox', 'Position', 
                             'Number_Object_Number']
                
                # Add user custom patterns
                custom_patterns = [p.strip() for p in self.custom_drop_patterns.value.split(',') if p.strip()]
                if custom_patterns:
                    blocklist.extend(custom_patterns)
                    print(f"- Added {len(custom_patterns)} custom patterns to blocklist.")
                
                # Rename BoundingBox columns to preserve them as Metadata if they exist
                # (BoundingBox is often technical, but sometimes user wants it as metadata? 
                #  User request implied efficient RAM. Storing unnecessary bbox is bad. 
                #  But let's stick to previous logic of 'Rename if not Metadata' to be safe, or just drop?)
                #  Let's keep the renaming logic but optimize it.
                
                columns = final_df.columns.tolist()
                bbox_cols = [c for c in columns if 'BoundingBox' in c and not c.startswith("Metadata_")]
                if bbox_cols:
                    rename_map = {c: f"Metadata_{c}" for c in bbox_cols}
                    final_df = final_df.rename(columns=rename_map)
                    print(f"- Renamed {len(bbox_cols)} BoundingBox columns to Metadata_ prefix.")
                    
                    # Update columns list
                    columns = final_df.columns.tolist()

                # Identify columns to drop (Technical Garbage)
                # Avoid dropping Metadata columns
                cols_to_drop = [c for c in columns if any(k in c for k in blocklist) and not c.startswith('Metadata_')]
                
                # Add specific custom columns
                custom_cols = [c.strip() for c in self.custom_drop_cols.value.split(',') if c.strip()]
                found_custom_cols = [c for c in custom_cols if c in columns]
                cols_to_drop.extend(found_custom_cols)

                # Deduplicate and Drop
                cols_to_drop = list(set(cols_to_drop))
                if cols_to_drop:
                    final_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
                    print(f"- Dropped {len(cols_to_drop)} blocklisted technical columns.")
                    gc.collect()
            
            # 4. Handle NaNs in Features
            # Identify Feature Cols (Numeric, non-Metadata)
            numeric_cols = final_df.select_dtypes(include=[np.number]).columns
            feature_cols = [c for c in numeric_cols if not c.startswith("Metadata_") and "WellID" not in c]
            
            if self.handle_nans.value != 'None':
                # Replace Inf with NaNs first
                final_df[feature_cols] = final_df[feature_cols].replace([np.inf, -np.inf], np.nan)
                
                if self.handle_nans.value == 'Drop Rows':
                    initial_rows = final_df.shape[0]
                    final_df = final_df.dropna(subset=feature_cols)
                    print(f"- Dropped {initial_rows - final_df.shape[0]} rows with NaNs in feature columns.")
                elif self.handle_nans.value == 'Fill with 0':
                    final_df[feature_cols] = final_df[feature_cols].fillna(0)
                    print("- Filled NaNs with 0.")
                
                gc.collect()

            # 5. Variance Threshold
            if self.variance_threshold_toggle.value:
                threshold = self.variance_threshold_value.value
                selector = VarianceThreshold(threshold=threshold)
                
                try:
                    # Use float32 for variance calc (should be fast)
                    X = final_df[feature_cols]
                    selector.fit(X)
                    support = selector.get_support()
                    
                    # Identify drops
                    features_to_drop = [f for f, kept in zip(feature_cols, support) if not kept]
                    
                    if features_to_drop:
                        final_df.drop(columns=features_to_drop, inplace=True)
                        # Update surviving list efficiently
                        feature_cols = [f for f, kept in zip(feature_cols, support) if kept]
                        print(f"- Dropped {len(features_to_drop)} low variance features (<={threshold}).")
                        gc.collect()
                        
                except ValueError as ve:
                    print(f"Warning: Variance Threshold failed (maybe no features left?): {ve}")

            # 6. Correlation Filter
            if self.correlation_threshold_toggle.value:
                threshold = self.correlation_threshold_value.value
                
                try:
                    # Correlation matrix can be huge. 
                    # If features > 5000, this might still OOM.
                    # Optimization: Use float32 checks again (already done).
                    if len(feature_cols) > 2000:
                        print(f"⚠️ Calculating correlation matrix for {len(feature_cols)} features. This may take memory...")
                    
                    # Compute correlation
                    corr_matrix = final_df[feature_cols].corr().abs()
                    
                    # Select upper triangle
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    
                    # Find columns with correlation > threshold
                    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
                    
                    if to_drop:
                        final_df.drop(columns=to_drop, inplace=True)
                        print(f"- Dropped {len(to_drop)} highly correlated features (>{threshold}).")
                        gc.collect()
                        
                except Exception as e:
                    print(f"❌ Error in correlation filter: {e}")
                    import traceback
                    traceback.print_exc()

            # Finalize
            ip.user_global_ns[output_name] = final_df
            print(f"\nSuccess! Filtered dataframe saved to '{output_name}'.")
            print(f"Final shape: {final_df.shape}")
            
            # Explicit cleanup
            del final_df
            gc.collect()

