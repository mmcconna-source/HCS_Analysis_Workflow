import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from typing import List, Optional

class WassersteinDistanceWidget:
    """
    Widget to calculate and visualize the Wasserstein distance between a Reference condition
    and multiple Test conditions across features.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # State
        self.fig = None
        self.ax = None
        self.results_df = None
        
        # UI Elements
        self._create_widgets()
        
    def _create_widgets(self):
        # 0. Mode Selection
        self.mode_toggle = widgets.ToggleButtons(
            options=['Reference vs All', 'All vs All'],
            description='Mode:',
            button_style=''
        )
        self.mode_toggle.observe(self._on_mode_change, names='value')

        # 0.5 Aggregation Metric
        self.metric_selector = widgets.Dropdown(
            options=['Mean', 'Median'],
            value='Mean',
            description='Metric:',
            layout=widgets.Layout(width='200px')
        )

        # 1. Group Selection
        self.group_cols_select = widgets.SelectMultiple(
            description='Group By:',
            rows=8,
            layout=widgets.Layout(height='160px', width='300px')
        )
        self.reload_cols()
        self.group_cols_select.observe(self._on_group_col_change, names='value')
        
        # 2. Reference Selection
        self.reference_dropdown = widgets.Dropdown(description='Reference:')
        self.ref_container = widgets.HBox([widgets.Label("Reference Group:"), self.reference_dropdown])
        
        # 3. Test Selection (Multi-select)
        self.test_group_select = widgets.SelectMultiple(description='Groups:', rows=10)
        
        # 4. Actions
        self.run_btn = widgets.Button(description='Calculate Distances', button_style='primary')
        self.run_btn.on_click(self.run_analysis)
        
        self.output_dir_text = widgets.Text(
            value='.',
            description='Output Dir:',
            placeholder='Path to save results'
        )
        
        self.save_btn = widgets.Button(description='Save Results to CSV', button_style='success', disabled=True)
        self.save_btn.on_click(self.save_results)
        
        self.output_area = widgets.Output()
        self.plot_output = widgets.Output()
        
        # Init figure
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        
        self._update_group_options()

    def reload_cols(self):
        """Populate initial group options."""
        options = [c for c in self.df.columns if c.startswith('Metadata_') or c in ['leiden', 'kmeans']]
        options = sorted(options)
        
        self.group_cols_select.options = options
        if options: 
            # Default select first option as tuple
            self.group_cols_select.value = (options[0],)

    def _on_group_col_change(self, change):
        self._update_group_options()

    def _on_mode_change(self, change):
        # Toggle Reference dropdown visibility
        if self.mode_toggle.value == 'All vs All':
            self.ref_container.layout.display = 'none'
            self.test_group_select.description = 'Select Groups to Compare:'
        else:
            self.ref_container.layout.display = 'flex'
            self.test_group_select.description = 'Test Groups:'

    def _get_group_series(self, cols):
        """Helper to get a single series representing the groups, handling multi-col."""
        if not cols: return pd.Series([], dtype=str)
        
        if len(cols) == 1:
            return self.df[cols[0]].astype(str)
        else:
            # Concatenate columns with underscore
            # df[c1].astype(str) + '_' + df[c2].astype(str) ...
            s = self.df[cols[0]].astype(str)
            for c in cols[1:]:
                s = s + '_' + self.df[c].astype(str)
            return s

    def _update_group_options(self):
        cols = self.group_cols_select.value
        if not cols:
            options = []
        else:
            # Check all cols exist
            valid_cols = [c for c in cols if c in self.df.columns]
            if len(valid_cols) != len(cols):
                options = []
            else:
                # Create combined series unique values
                combined = self._get_group_series(valid_cols)
                options = sorted(combined.unique())
            
        self.reference_dropdown.options = options
        self.test_group_select.options = options
        
        # Default selection
        if len(options) > 0: self.reference_dropdown.value = options[0]
        # Multi-select value
        if len(options) > 1: self.test_group_select.value = (options[1],)

    def _get_feature_columns(self):
        """Exclude Metadata_ columns and coordinate columns."""
        exclude_cols = [c for c in self.df.columns if c.startswith('Metadata_')]
        exclude_cols += [
            'UMAP1', 'UMAP2', 'ImageNumber', 'ObjectNumber', 'leiden', 'kmeans'
        ]
        # Heuristic: exclude columns that are likely not features (e.g. string columns)
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        # Ensure numeric
        numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols

    def run_analysis(self, btn):
        mode = self.mode_toggle.value
        if mode == 'Reference vs All':
            self.run_reference_analysis()
        else:
            self.run_matrix_analysis()
        
        # Enable save button if we have results
        if self.results_df is not None and not self.results_df.empty:
            self.save_btn.disabled = False

    def save_results(self, btn):
        if self.results_df is None or self.results_df.empty:
            return
            
        import datetime
        import os
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "matrix" if self.mode_toggle.value == 'All vs All' else "ref_vs_all"
        filename = f"wasserstein_results_{mode}_{timestamp}.csv"
        
        out_dir = self.output_dir_text.value.strip()
        if not out_dir: out_dir = '.'
        
        # Create dir if not exists
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except Exception as e:
                with self.output_area:
                    print(f"❌ Error creating directory {out_dir}: {e}")
                return

        filepath = os.path.join(out_dir, filename)
        
        try:
            self.results_df.to_csv(filepath)
            with self.output_area:
                print(f"✅ Results saved to {filepath}")
        except Exception as e:
            with self.output_area:
                print(f"❌ Error saving CSV: {e}")

    def run_matrix_analysis(self):
        cols = self.group_cols_select.value
        selected_groups = list(self.test_group_select.value)
        metric = self.metric_selector.value
        
        if not cols or len(selected_groups) < 2:
            with self.output_area:
                clear_output()
                print("Please select at least two groups for Matrix comparison.")
            return
            
        with self.output_area:
            clear_output()
            print(f"Calculating All-vs-All Wasserstein Matrix ({metric})...")
            print(f"Groups ({len(selected_groups)}): {selected_groups}")
            
            feature_cols = self._get_feature_columns()
            col_series = self._get_group_series(cols)
            
            # Pre-fetch data for all groups to avoid repeated indexing
            group_data = {}
            for g in selected_groups:
                mask = col_series == str(g)
                data = self.df.loc[mask, feature_cols]
                if data.empty:
                    print(f"Warning: No data for {g}")
                    continue
                # Pre-convert to numeric/clean
                cleaned_features = {}
                for f in feature_cols:
                     vals = pd.to_numeric(data[f], errors='coerce').values
                     cleaned_features[f] = vals[np.isfinite(vals)]
                group_data[g] = cleaned_features

            # valid groups 
            valid_groups = list(group_data.keys())
            
            # Matrix to store Mean/Median Wasserstein Distance
            # Rows/Cols = Groups
            matrix = pd.DataFrame(index=valid_groups, columns=valid_groups, dtype=float)
            
            # Prepare pairs
            import itertools
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            pairs = list(itertools.combinations_with_replacement(valid_groups, 2))
            
            # Helper function for parallel execution
            def calc_pair_dist(p):
                g1, g2 = p
                if g1 == g2:
                    return (g1, g2, 0.0)
                
                dists = []
                for f in feature_cols:
                    u = group_data[g1][f]
                    v = group_data[g2][f]
                    if len(u) == 0 or len(v) == 0: continue
                    d = wasserstein_distance(u, v)
                    dists.append(d)
                
                if dists:
                    agg = np.median(dists) if metric == 'Median' else np.mean(dists)
                    return (g1, g2, agg)
                else:
                    return (g1, g2, np.nan)

            # Execution
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(calc_pair_dist, p): p for p in pairs}
                for future in as_completed(futures):
                    g1, g2, val = future.result()
                    matrix.loc[g1, g2] = val
                    matrix.loc[g2, g1] = val

            self.results_df = matrix
            print("Matrix Calculation Complete.")
            self.plot_matrix_results()

    def plot_matrix_results(self):
        if self.results_df is None or self.results_df.empty: return
        self.ax.clear()
        self.fig.clf()
        metric = self.metric_selector.value
        
        # Use clustermap if possible, but it creates its own figure
        # So we can't easily embed it into self.fig. 
        # We will use sns.heatmap on self.fig for now to keep it simple in the widget widget structure
        # Or recalculate linkage and sort our matrix to look like a cluster map
        
        self.ax = self.fig.add_subplot(111)
        
        sns.heatmap(self.results_df, cmap='viridis', ax=self.ax, annot=False, 
                    cbar_kws={'label': f'{metric} Wasserstein Distance'})
        
        self.ax.set_title(f"Pairwise {metric} Wasserstein Distance")
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right")
        
        self.fig.tight_layout()
        
        with self.plot_output:
            clear_output(wait=True)
            display(self.fig)

    def run_reference_analysis(self):
        cols = self.group_cols_select.value
        ref_group = self.reference_dropdown.value
        test_groups = list(self.test_group_select.value)
        
        if not cols or not ref_group or not test_groups:
            with self.output_area: 
                clear_output()
                print("Please select a grouping column, a reference, and at least one test group.")
            return
            
        with self.output_area:
            clear_output()
            print(f"Calculating Wasserstein Distances...")
            print(f"Reference Group: {ref_group}")
            print(f"Test Groups: {test_groups}")
            
            feature_cols = self._get_feature_columns()
            col_series = self._get_group_series(cols)
            
            # --- Prepare Data (Parallel Optimized) ---
            # 1. Clean Reference Data once
            ref_mask = col_series == str(ref_group)
            ref_raw = self.df.loc[ref_mask, feature_cols]
            
            if ref_raw.empty:
                print(f"Error: No data found for Reference group '{ref_group}'.")
                return

            ref_data = {}
            for f in feature_cols:
                vals = pd.to_numeric(ref_raw[f], errors='coerce').values
                ref_data[f] = vals[np.isfinite(vals)]

            # 2. Helper to process one test group
            def process_test_group(tg):
                if str(tg) == str(ref_group): return (tg, None)
                
                t_mask = col_series == str(tg)
                t_raw = self.df.loc[t_mask, feature_cols]
                
                if t_raw.empty: return (tg, {}) # Empty dict signals no data
                
                t_dists = {}
                for f in feature_cols:
                    u = ref_data[f]
                    # Clean test feature
                    v_vals = pd.to_numeric(t_raw[f], errors='coerce').values
                    v = v_vals[np.isfinite(v_vals)]
                    
                    if len(u) == 0 or len(v) == 0:
                        dist = np.nan
                    else:
                        dist = wasserstein_distance(u, v)
                    t_dists[f] = dist
                return (tg, t_dists)

            # 3. Parallel Execution
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            all_results = {}
            # Filter out self-compare from list to save threads
            real_test_groups = [g for g in test_groups if str(g) != str(ref_group)]
            
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_test_group, g) for g in real_test_groups]
                
                for future in as_completed(futures):
                    group_name, dists = future.result()
                    if dists is None: continue # was self or errored
                    if not dists: 
                        print(f"Warning: No data found for Test group '{group_name}'.")
                        continue
                    all_results[group_name] = dists
            
            if not all_results:
                 print("Error: No valid comparisons made (or only self-comparison was selected).")
                 return

            # Convert to DataFrame: Rows = Features, Columns = Test Groups
            self.results_df = pd.DataFrame(all_results)
            
            # Calculate Total Distance per Feature to sort them
            self.results_df['Total_Distance'] = self.results_df.sum(axis=1)
            self.results_df = self.results_df.sort_values('Total_Distance', ascending=False)
            
            # Drop the helper column for plotting
            results_for_plot = self.results_df.drop(columns=['Total_Distance'])
            self.results_df = results_for_plot # Update state
            
            print("Analysis Complete.")
            self.plot_results()

    def select_all_test_groups(self, btn):
        options = list(self.test_group_select.options)
        self.test_group_select.value = tuple(options)

    def plot_results(self):
        if self.results_df is None or self.results_df.empty: return
        
        self.ax.clear()
        metric = self.metric_selector.value

        # Calculate Mean/Median Distance per Condition for summary plot
        if metric == 'Median':
            condition_dist = self.results_df.median(axis=0).sort_values(ascending=False)
        else:
            condition_dist = self.results_df.mean(axis=0).sort_values(ascending=False)
        
        # We'll do two plots: Heatmap (Left) and Summary Bar Plot (Right)
        # Re-setup figure with gridspec
        self.fig.clf()
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax_heatmap = self.fig.add_subplot(gs[0])
        ax_bar = self.fig.add_subplot(gs[1])
        
        # --- Heatmap ---
        # We might have too many features to show all in a heatmap if we don't filter.
        # Let's show top 50 features max for readability
        n_features_to_show = min(50, len(self.results_df))
        plot_data = self.results_df.head(n_features_to_show)
        
        sns.heatmap(plot_data, cmap='viridis', ax=ax_heatmap, 
                    cbar_kws={'label': 'Wasserstein Distance'})
        
        ax_heatmap.set_title(f"Distance relative to Ref: {self.reference_dropdown.value}")
        ax_heatmap.set_xlabel("Test Conditions")
        ax_heatmap.set_ylabel(f"Features (Top {n_features_to_show})")
        
        # Rotate x labels if needed
        plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # --- Bar Plot ---
        colors = plt.cm.viridis(np.linspace(0, 1, len(condition_dist)))
        condition_dist.plot(kind='barh', ax=ax_bar, color=colors)
        ax_bar.set_title(f"{metric} Wasserstein Distance (All Features)")
        ax_bar.set_xlabel(f"{metric} Distance")
        ax_bar.invert_yaxis() # Highest distance on top
        
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
        # Layout
        
        # Add Select All Button
        select_all_btn = widgets.Button(description='Select All Conditions', icon='check-square')
        select_all_btn.on_click(self.select_all_test_groups)

        controls = widgets.VBox([
            widgets.HTML("<h3>Wasserstein Distance Analysis</h3>"),
            widgets.HBox([self.mode_toggle, self.metric_selector]),
            widgets.HTML("<hr>"),
            self.group_cols_select,
            
            self.ref_container,
            
            # widgets.HTML("<b>Select Test Groups (Comparisons):</b>"),
            select_all_btn,
            self.test_group_select,
            
            widgets.HTML("<hr>"),
            self.output_dir_text,
            widgets.HBox([self.run_btn, self.save_btn]),
            self.output_area
        ])
        
        # Ensure plot output is visible
        layout = widgets.HBox([controls, self.plot_output])
        display(layout)
