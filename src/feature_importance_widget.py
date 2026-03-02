import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List

class FeatureImportanceWidget:
    """
    Widget to identify features driving separation between two groups of clusters 
    using a Random Forest classifier.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # State
        self.fig = None
        self.ax = None
        self.feature_importances = None
        
        # UI Elements
        self._create_widgets()
        
    def _create_widgets(self):
        # 1. Group Selection
        self.group_col_dropdown = widgets.Dropdown(description='Group By:')
        self.reload_cols()
        self.group_col_dropdown.observe(self._on_group_col_change, names='value')
        
        # 2. Comparison Selection
        self.group_a_select = widgets.SelectMultiple(description='Group A:', rows=6)
        self.group_b_select = widgets.SelectMultiple(description='Group B:', rows=6)
        
        # 3. Model Parameters
        self.model_dropdown = widgets.Dropdown(description='Model:', options=['Random Forest', 'XGBoost'], value='Random Forest')
        self.n_estimators_slider = widgets.IntSlider(value=100, min=10, max=500, step=10, description='Trees:')
        
        # 4. Actions
        self.run_btn = widgets.Button(description='Run Analysis', button_style='primary')
        self.run_btn.on_click(self.run_analysis)
        
        self.output_area = widgets.Output()
        self.plot_output = widgets.Output()
        
        # Init figure
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        
        self._update_group_options()

    def reload_cols(self):
        """Populate initial group options."""
        options = [c for c in self.df.columns if c.startswith('Metadata_') or c in ['leiden', 'kmeans']]
        # Sort but ensure 'leiden' is first default if exists
        options = sorted(options)
        
        if 'leiden' in options: default = 'leiden'
        elif 'kmeans' in options: default = 'kmeans'
        elif options: default = options[0]
        else: default = None
            
        self.group_col_dropdown.options = options
        if default: self.group_col_dropdown.value = default

    def _on_group_col_change(self, change):
        self._update_group_options()

    def _update_group_options(self):
        col = self.group_col_dropdown.value
        if not col or col not in self.df.columns:
            options = []
        else:
            options = sorted([str(x) for x in self.df[col].dropna().unique()])
            
        self.group_a_select.options = options
        self.group_b_select.options = options
        
        # Default selection: First item for A, Second (if exists) for B
        if len(options) > 0: self.group_a_select.value = (options[0],)
        if len(options) > 1: self.group_b_select.value = (options[1],)

    def _get_feature_columns(self):
        """Exclude Metadata_ columns and coordinate columns."""
        exclude_cols = [c for c in self.df.columns if c.startswith('Metadata_')]
        exclude_cols += [
            'UMAP1', 'UMAP2', 'ImageNumber', 'ObjectNumber', 'leiden', 'kmeans'
        ]
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        # Ensure numeric
        numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols

    def run_analysis(self, btn):
        col = self.group_col_dropdown.value
        group_a = list(self.group_a_select.value)
        group_b = list(self.group_b_select.value)
        
        if not col or not group_a or not group_b:
            with self.output_area: print("Please select a grouping column and clusters for both groups.")
            return
            
        # Ensure no overlap (optional, but good practice)
        overlap = set(group_a).intersection(set(group_b))
        if overlap:
             with self.output_area: print(f"Warning: Groups overlap ({overlap}). Intersection will be treated as Group B in binary labels (0/1), or better: excluded.")
        
        with self.output_area:
            clear_output()
            print(f"Comparing Group A {group_a} vs Group B {group_b}...")
            
            # Prepare Data
            feature_cols = self._get_feature_columns()
            
            # Filter mask
            # Convert col to string for matching options which are strings
            col_series = self.df[col].astype(str)
            
            mask_a = col_series.isin(group_a)
            mask_b = col_series.isin(group_b)
            
            # Exclusive mask (if an item is in both selected lists, what do? Let's check pure overlap)
            # Actually, let's just create a subset
            df_a = self.df[mask_a].copy()
            df_a['Target'] = 0 # Group A
            
            df_b = self.df[mask_b].copy()
            df_b['Target'] = 1 # Group B
            
            # Handle collision if same row selected in both (e.g. same value in options list)
            # Since options come from unique values of ONE column, overlap is impossible unless user selected same value in both boxes.
            if set(group_a).intersection(set(group_b)):
                 print("Error: Same cluster selected in both groups.")
                 return

            combined_df = pd.concat([df_a, df_b])
            
            if combined_df.empty:
                print("Error: No data found for selected groups.")
                return
                
            X = combined_df[feature_cols].values
            y = combined_df['Target'].values
            
            print(f"Training {self.model_dropdown.value} on {len(combined_df)} cells ({len(feature_cols)} features)...")
            
            if self.model_dropdown.value == 'Random Forest':
                clf = RandomForestClassifier(
                    n_estimators=self.n_estimators_slider.value,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                clf.fit(X, y)
                importances = clf.feature_importances_
                
            elif self.model_dropdown.value == 'XGBoost':
                try:
                    from xgboost import XGBClassifier
                except ImportError:
                     print("Error: 'xgboost' not installed. Please install it (pip install xgboost).")
                     return

                clf = XGBClassifier(
                    n_estimators=self.n_estimators_slider.value,
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=(len(y)-sum(y))/sum(y) if sum(y) > 0 else 1, # Balance classes
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                clf.fit(X, y)
                importances = clf.feature_importances_
            
            # Extract Importances
            indices = np.argsort(importances)[::-1]
            
            # Store results
            result_data = []
            for f in range(min(20, len(feature_cols))): # Top 20
                idx = indices[f]
                result_data.append({
                    'Feature': feature_cols[idx],
                    'Importance': importances[idx]
                })
            
            self.results_df = pd.DataFrame(result_data)
            
            print("Analysis Complete.")
            self.plot_results(group_a, group_b)

    def plot_results(self, group_a, group_b):
        if self.results_df is None or self.results_df.empty: return
        
        self.ax.clear()
        
        # Plot
        y_pos = np.arange(len(self.results_df))
        self.ax.barh(y_pos, self.results_df['Importance'], align='center', color='skyblue')
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(self.results_df['Feature'])
        self.ax.invert_yaxis()  # labels read top-to-bottom
        self.ax.set_xlabel('Importance')
        self.ax.set_title(f"Top Discriminative Features\n{group_a} vs {group_b}")
        
        self.fig.tight_layout()
        
        # Display
        with self.plot_output:
            clear_output(wait=True)
            import matplotlib
            backend = matplotlib.get_backend().lower()
            if 'ipympl' in backend or 'widget' in backend:
                 display(self.fig.canvas)
            else:
                 display(self.fig)

    def display(self):
        col1 = widgets.VBox([
            widgets.HTML("<h3>Comparison Setup</h3>"),
            self.group_col_dropdown,
            widgets.HBox([self.group_a_select, self.group_b_select]),
            widgets.HTML("<hr>"),
            self.model_dropdown,
            self.n_estimators_slider,
            self.run_btn,
            self.output_area
        ])
        
        col2 = widgets.VBox([
            self.plot_output
        ])
        
        display(widgets.HBox([col1, col2]))
