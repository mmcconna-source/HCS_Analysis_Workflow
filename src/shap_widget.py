import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import os
import seaborn as sns

class SHAPWidget:
    """
    Widget for Interactive SHAP Analysis on XGBoost Models.
    Robust implementation focusing on data cleaning and standard explainer usage.
    """
    def __init__(self, model, X, class_names=None):
        """
        Args:
            model: Trained XGBoost model object (XGBClassifier or Booster) or path to .json/.ubj/.model.
            X: DataFrame containing features to explain. (Can include metadata; will be filtered).
            class_names: List of class names (optional).
        """
        self.model_input = model
        self.X_input = X
        self.class_names = class_names
        
        # Internal State
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X_display = None # Clean feature DataFrame used for SHAP
        
        # Setup UI
        self._create_widgets()
        
    def _create_widgets(self):
        # -- Header & Controls --
        self.header = widgets.HTML("<h3>SHAP Model Interpretation</h3>")
        
        self.run_btn = widgets.Button(
            description='Run SHAP Analysis', 
            button_style='primary',
            icon='chart-bar'
        )
        self.run_btn.on_click(self.run_analysis)
        
        self.status_output = widgets.Output(layout={'border': '1px solid #eee', 'padding': '4px', 'margin-top': '5px'})
        
        # -- Tabs --
        self.summary_out = widgets.Output() # Beeswarm
        self.bar_out = widgets.Output()     # Importance Bars
        self.dep_out = widgets.Output()     # Dependence Plot
        
        # Dependence Controls
        self.feature_picker = widgets.Dropdown(description='Feature:')
        self.feature_picker.observe(self._on_feature_change, names='value')
        
        self.dep_plot_area = widgets.Output()
        
        # Layout Dependence Tab
        dep_layout = widgets.VBox([
            widgets.HBox([self.feature_picker]),
            self.dep_plot_area
        ])
        
        # Main Tab Container
        self.tabs = widgets.Tab(children=[self.summary_out, self.bar_out, dep_layout])
        self.tabs.set_title(0, 'Summary (Beeswarm)')
        self.tabs.set_title(1, 'Feature Importance')
        self.tabs.set_title(2, 'Dependence')
        
        # Main Layout
        self.container = widgets.VBox([
            self.header,
            widgets.HBox([self.run_btn]),
            self.status_output,
            widgets.HTML("<hr style='margin: 10px 0;'>"),
            self.tabs
        ])
        
    def log(self, msg, error=False):
        color = "red" if error else "black"
        with self.status_output:
            print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")
            
    def clear_log(self):
        self.status_output.clear_output()
        
    def run_analysis(self, change):
        self.clear_log()
        self.run_btn.disabled = True
        self.log("Starting analysis...")
        
        try:
            # 1. Load/Validate Model
            if not self._load_model():
                return
                
            # 2. Prepare Data (Strict Cleaning)
            if not self._prepare_data():
                return
                
            # 3. Compute SHAP
            if not self._compute_shap():
                return
                
            # 4. Render Plots
            self._render_summary()
            self._render_bar()
            self._init_dependence()
            
            self.log("Analysis complete!")
            
        except Exception as e:
            self.log(f"Critical Error: {str(e)}", error=True)
            with self.status_output:
                import traceback
                traceback.print_exc()
        finally:
            self.run_btn.disabled = False
            
    def _load_model(self):
        """Loads model from file or uses object. Handles XGBClassifier vs Booster."""
        try:
            if isinstance(self.model_input, str):
                path = self.model_input
                if not os.path.exists(path):
                    self.log(f"File not found: {path}", error=True)
                    return False
                
                self.log(f"Loading model from {path}...")
                self.model = xgb.XGBClassifier()
                try:
                    self.model.load_model(path)
                except Exception as load_err:
                     self.log(f"Standard load failed: {load_err}. Attempting JSON patch...", error=True)
                     # Attempt patch if load failed (though usually load succeeds, SHAP fails. 
                     # But sometimes load fails if format is weird. 
                     # Actually the error reported was during explainer creation, so load likely worked.)
                     raise load_err
            else:
                self.model = self.model_input
                
            # Validation: Try to get booster to ensure it's a valid XGB object
            if hasattr(self.model, 'get_booster'):
                self.model.get_booster() 
            return True
            
        except Exception as e:
            self.log(f"Model load failed: {e}", error=True)
            return False
            
    def _prepare_data(self):
        """Removes non-numeric columns and validates data."""
        if self.X_input is None:
            self.log("No dataframe provided.", error=True)
            return False
            
        # Select numeric types only
        df_num = self.X_input.select_dtypes(include=[np.number])
        
        # Convention: Drop 'Metadata_' and 'Image_' columns even if numeric (e.g. IDs)
        # Also drop standard irrelevant columns often found in these pipelines
        drop_patterns = ['Metadata_', 'Image_', 'FileName_', 'PathName_']
        cols = [c for c in df_num.columns if not any(c.startswith(p) for p in drop_patterns)]
        
        # Also drop specific known tracking columns if present
        drop_exact = {'object_number', 'image_number', 'well_id', 'field_id'}
        cols = [c for c in cols if c.lower() not in drop_exact]
        
        self.X_display = df_num[cols]
        
        if self.X_display.empty:
            self.log("No valid numeric features found after filtering.", error=True)
            return False
            
        self.log(f"Selected {len(self.X_display.columns)} features for analysis.")
        return True

    def _compute_shap(self):
        """Initializes explainer and calculates values."""
        self.log("Calculating SHAP values (this may take a moment)...")
        
        try:
            model_to_explain = self.model
            
            # --- Attempt 1: TreeExplainer Direct ---
            try:
                self.explainer = shap.TreeExplainer(model_to_explain)
                # Test logic
                self.explainer.shap_values(self.X_display.iloc[:2])
            except Exception as e:
                self.log(f"TreeExplainer direct failed: {e}. Attempting patch...")
                
                # --- Attempt 2: JSON Patch for base_score ---
                try:
                    import tempfile
                    import json
                    import re
                    
                    # 1. Save booster to JSON
                    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp:
                        tmp_path = tmp.name
                    
                    booster = model_to_explain.get_booster()
                    booster.save_model(tmp_path)
                    
                    # 2. Read and Patch
                    with open(tmp_path, 'r') as f:
                        content = f.read()
                        
                    # Fix base_score: ["0.5"] -> "0.5" or similar
                    # The error 'could not convert string to float: [4.875E-1]' implies it's a list string
                    pattern = r'"base_score"\s*:\s*\[\s*"([^"]+)"\s*\]'
                    if re.search(pattern, content):
                        self.log("Detected malformed base_score in JSON. Patching...")
                        new_content = re.sub(pattern, r'"base_score":"\1"', content)
                        
                        with open(tmp_path, 'w') as f:
                            f.write(new_content)
                            
                        # 3. Reload into fresh model
                        self.model = xgb.XGBClassifier()
                        self.model.load_model(tmp_path)
                        self.explainer = shap.TreeExplainer(self.model)
                    else:
                        self.log("No base_score pattern found to patch.")
                        raise e # Re-raise if we couldn't fix it
                        
                    # Cleanup
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
                        
                except Exception as patch_err:
                     self.log(f"Patch failed: {patch_err}")
                     raise e # Trigger fallback
            
            # Calculate values
            self.shap_values = self.explainer.shap_values(self.X_display)
            return True
            
        except Exception as e:
            self.log(f"TreeExplainer failed: {e}", error=True)
            
            # Fallback suggestion
            self.log("Attempting fallback to KernelExplainer (slow)...")
            try:
                # Subsample background for speed
                bg = shap.sample(self.X_display, 50)
                
                # Wrapper for probability output
                def predict_fn(data):
                    if isinstance(data, np.ndarray):
                        col_names = self.X_display.columns
                        data = pd.DataFrame(data, columns=col_names)
                    
                    if hasattr(self.model, 'predict_proba'):
                        return self.model.predict_proba(data)
                    else:
                        return self.model.predict(data) # Fallback if no proba
                
                self.explainer = shap.KernelExplainer(predict_fn, bg)
                self.shap_values = self.explainer.shap_values(self.X_display)
                return True
            except Exception as k_e:
                self.log(f"Kernel fallback also failed: {k_e}", error=True)
                return False

    def _render_summary(self):
        with self.summary_out:
            clear_output(wait=True)
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Handle multiclass list output
                vals = self.shap_values
                if isinstance(vals, list):
                    # For summary plot, we can pass the list directly usually, 
                    # but for binary it's often cleaner to just show positive class
                    if len(vals) == 2:
                        vals = vals[1]
                    else:
                         vals = vals[0] # Multiclass or weird binary, pick first
                
                shap.summary_plot(vals, self.X_display, show=False)
                plt.tight_layout()
                display(fig)
            except Exception as e:
                print(f"Error plotting summary: {e}")

    def _render_bar(self):
        with self.bar_out:
            clear_output(wait=True)
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Again handle list vs array
                vals = self.shap_values
                if isinstance(vals, list):
                    if len(vals) == 2:
                        vals = vals[1]
                    else:
                        vals = vals[0]
                
                shap.summary_plot(vals, self.X_display, plot_type="bar", show=False)
                plt.tight_layout()
                display(fig)
            except Exception as e:
                print(f"Error plotting bar: {e}")

    def _init_dependence(self):
        # Populate dropdown
        self.feature_picker.options = self.X_display.columns.tolist()
        if len(self.X_display.columns) > 0:
            self.feature_picker.value = self.X_display.columns[0]
            # Trigger first plot
            self._on_feature_change({'new': self.feature_picker.value})
            
    def _on_feature_change(self, change):
        if not change['new'] or self.shap_values is None: 
            return
            
        feature = change['new']
        
        with self.dep_plot_area:
            clear_output(wait=True)
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Dependence plot requires a single matrix of values [n_samples, n_features]
                vals = self.shap_values
                if isinstance(vals, list):
                    # Use positive class or first class
                    # vals is list of [N, M] arrays
                    vals = vals[1] if len(vals) >= 2 else vals[0]
                
                # vals should now be (N, M)
                # self.X_display is (N, M) DataFrame
                
                shap.dependence_plot(feature, vals, self.X_display, show=False, ax=ax)
                plt.tight_layout()
                display(fig)
            except Exception as e:
                print(f"Error plotting dependence: {e}")

    def display(self):
        display(self.container)
