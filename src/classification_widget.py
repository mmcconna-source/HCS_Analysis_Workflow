import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import random

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.tile_extraction import NotebookConfig, extract_multichannel_tile, create_rgb_composite, TileExtractionError
from src.channel_mapping_widget import ChannelMappingWidget

class ClassificationWidget:
    """
    Widget for Interactive Classification Workflow.
    1. Configure Tile Appearance
    2. Manually Annotate Tiles
    3. Train XGBoost Model
    4. Predict on Full Dataset
    """
    def __init__(self, df: pd.DataFrame, config: NotebookConfig = None, class_names=['Positive', 'Negative']):
        self.df = df # Main dataset (operate in-place)
        
        # If config not provided, try to infer or create default
        if config is None:
             self.config = NotebookConfig(
                 csv_path='.', 
                 image_base_path='.',
                 channel_names=['C1','C2','C3','C4','C5'] 
             )
        else:
             self.config = config
             
        self.class_names = class_names
        
        # State
        self.annotations = {} # {index: label}
        self.annotation_queue = []
        self.current_annotation_index = -1
        
        self.model = None
        self.le = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        
        self.cached_tiles = {} # {index: tile_array}
        
        # UI Elements
        self._create_widgets()
        
    def _create_widgets(self):
        # --- TAB 1: ANNOTATION ---
        
        # 1. Tile Configuration (Embedded ChannelMappingWidget)
        # Try to sample tiles for preview from 5 unique images
        sample_tiles = []
        try:
            if not self.df.empty and hasattr(self.config, 'image_base_path'):
                 # Create a unique key for image identity (Well + Field)
                 # Adjust column names if needed based on config
                 well_col = self.config.well_column
                 field_col = self.config.field_column
                 
                 if well_col in self.df.columns and field_col in self.df.columns:
                     # Include plate column for multi-plate datasets
                     group_cols = [well_col, field_col]
                     plate_col = getattr(self.config, 'plate_column', 'Metadata_PlateID')
                     if plate_col in self.df.columns:
                         group_cols.append(plate_col)
                         
                     unique_images = self.df[group_cols].drop_duplicates()
                     
                     # Sample up to 5 unique images
                     n_images = min(5, len(unique_images))
                     if n_images > 0:
                         sampled_images = unique_images.sample(n_images)
                         
                         for _, img_row in sampled_images.iterrows():
                             # Get one cell from this image
                             cell_mask = (self.df[well_col] == img_row[well_col]) & (self.df[field_col] == img_row[field_col])
                             if plate_col in self.df.columns:
                                 cell_mask = cell_mask & (self.df[plate_col] == img_row[plate_col])
                             potential_cells = self.df[cell_mask]
                             
                             if not potential_cells.empty:
                                 # Pick random cell from this image
                                 cell_row = potential_cells.sample(1).iloc[0]
                                 try:
                                     tile = extract_multichannel_tile(cell_row, self.config)
                                     sample_tiles.append(tile)
                                 except Exception:
                                     pass
                 else:
                     # Fallback to random sampling if columns missing
                     sample_indices = random.sample(list(self.df.index), min(5, len(self.df)))
                     for idx in sample_indices:
                         try:
                             row = self.df.loc[idx]
                             tile = extract_multichannel_tile(row, self.config)
                             sample_tiles.append(tile)
                         except Exception:
                             pass
                             
        except Exception as e:
            print(f"Warning: Could not load preview tiles: {e}")
            
        self.channel_widget = ChannelMappingWidget(self.config, sample_tiles=sample_tiles)
        
        # 2. Sampling Controls
        self.n_samples_input = widgets.IntText(value=50, description='Count:', layout=widgets.Layout(width='150px'))
        
        # New Filter Controls
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.filter_feature_dropdown = widgets.Combobox(
            options=['None'] + num_cols, 
            value='None', 
            description='Filter By:', 
            ensure_option=True, 
            layout=widgets.Layout(width='300px'),
            placeholder='Type to search features...'
        )
        self.filter_operator_dropdown = widgets.Dropdown(options=['>', '<', '>=', '<='], value='>', layout=widgets.Layout(width='60px'))
        self.filter_cutoff_input = widgets.FloatText(value=0.0, layout=widgets.Layout(width='100px'))
        self.filter_box = widgets.HBox([self.filter_feature_dropdown, self.filter_operator_dropdown, self.filter_cutoff_input])

        self.start_annotation_btn = widgets.Button(description='Start Annotation', button_style='primary')
        self.start_annotation_btn.on_click(self.start_annotation)
        
        self.reset_annotation_btn = widgets.Button(description='Reset All Labels', button_style='warning')
        self.reset_annotation_btn.on_click(self.reset_annotations)
        
        # 3. Annotation Interface
        self.annotation_output = widgets.Output() # For the image
        self.class_buttons = []
        
        # Build shortcut instructions
        shortcuts = []
        for i, cls in enumerate(self.class_names):
            key = str(i + 1)
            btn = widgets.Button(description=f"{cls} ({key})")
            btn.on_click(lambda b, c=cls: self.annotate_current(c))
            self.class_buttons.append(btn)
            shortcuts.append(f"'{key}'={cls}")
            
        self.skip_btn = widgets.Button(description='Skip (s)')
        self.skip_btn.on_click(self.skip_current)
        shortcuts.append("'s'=Skip")
        
        # Keyboard listener text box
        self.keyboard_listener = widgets.Text(placeholder="Click here to use keyboard: " + ", ".join(shortcuts), layout=widgets.Layout(width='400px'))
        self.keyboard_listener.observe(self._handle_keypress, names='value')
        
        self.progress_label = widgets.Label(value="Progress: 0 / 0")
        
        # Layout Tab 1
        anno_controls = widgets.VBox([
            widgets.HTML("<h4>1. Configure & Sample</h4>"),
            self.channel_widget.widget_container,
            widgets.HTML("<b>Filter cells for annotation:</b>"),
            self.filter_box,
            widgets.HBox([self.n_samples_input, self.start_annotation_btn, self.reset_annotation_btn]),
            widgets.HTML("<hr><h4>2. Manual Labeling</h4>"),
            self.annotation_output,
            widgets.HBox(self.class_buttons + [self.skip_btn]),
            self.keyboard_listener,
            self.progress_label
        ])
        
        # --- TAB 2: TRAINING ---
        self.test_size_slider = widgets.FloatSlider(value=0.2, min=0.1, max=0.5, step=0.05, description='Test Size:')
        self.train_btn = widgets.Button(description='Train Model', button_style='success')
        self.train_btn.on_click(self.train_model)
        
        self.train_log_output = widgets.Output(layout={'border': '1px solid #ddd', 'height': '150px', 'overflow': 'auto'})
        self.result_output = widgets.Output()
        
        # Save Model UI
        self.save_model_input = widgets.Text(description='Model Name:', value='xgb_model.ubj')
        self.save_model_btn = widgets.Button(description='Save Model')
        self.save_model_btn.on_click(self.save_trained_model)
        
        # Figure for metrics
        with plt.ioff():
            self.fig_metrics, self.ax_metrics = plt.subplots(figsize=(5, 4))
        self.fig_metrics.canvas.header_visible = False
        self.fig_metrics.canvas.footer_visible = False

        train_panel = widgets.VBox([
            widgets.HTML("<h3>Train Classifier</h3>"),
            widgets.HTML("<p>Train an XGBoost model using your manual annotations.</p>"),
            widgets.HBox([self.test_size_slider, self.train_btn]),
            self.train_log_output,
            self.result_output,
            widgets.HTML("<hr>"),
            widgets.HBox([self.save_model_input, self.save_model_btn])
        ])
        
        # --- TAB 3: PREDICTION ---
        self.predict_btn = widgets.Button(description='Predict All Cells', button_style='info')
        self.predict_btn.on_click(self.predict_all)
        
        self.save_name_input = widgets.Text(description='File Suffix:', value='_classified.csv')
        self.save_btn = widgets.Button(description='Save Predictions')
        self.save_btn.on_click(self.save_predictions)
        
        self.predict_log_output = widgets.Output()
        
        predict_panel = widgets.VBox([
            widgets.HTML("<h3>Apply Model</h3>"),
            self.predict_btn,
            self.predict_log_output,
            widgets.HTML("<hr>"),
            widgets.HBox([self.save_name_input, self.save_btn])
        ])

        # Main Tabs
        self.tabs = widgets.Tab(children=[anno_controls, train_panel, predict_panel])
        self.tabs.set_title(0, 'Annotate')
        self.tabs.set_title(1, 'Train')
        self.tabs.set_title(2, 'Predict')

    def log_train(self, msg):
        with self.train_log_output:
            print(msg)

    def log_predict(self, msg):
        with self.predict_log_output:
            print(msg)
            
    def save_trained_model(self, btn):
        if self.model is None:
            self.log_train("Error: Train a model first.")
            return
            
        filename = self.save_model_input.value
        try:
            # XGBoost save using JSON format usually
            self.model.save_model(filename)
            self.log_train(f"Model saved to {filename}")
        except Exception as e:
            self.log_train(f"Error saving model: {e}")
            
    # --- ANNOTATION LOGIC ---
        
    def start_annotation(self, btn):
        """Sample random un-annotated cells and start the queue."""
        n = self.n_samples_input.value
        
        # Apply Filters First
        filter_col = self.filter_feature_dropdown.value
        if filter_col != 'None' and filter_col in self.df.columns:
            op = self.filter_operator_dropdown.value
            val = self.filter_cutoff_input.value
            
            if op == '>':
                valid_mask = self.df[filter_col] > val
            elif op == '<':
                valid_mask = self.df[filter_col] < val
            elif op == '>=':
                valid_mask = self.df[filter_col] >= val
            elif op == '<=':
                valid_mask = self.df[filter_col] <= val
                
            filtered_df = self.df[valid_mask]
            with self.annotation_output: 
                 print(f"Applied filter: {filter_col} {op} {val}. Found {len(filtered_df)} cells.")
        else:
            filtered_df = self.df
            
        # Get indices not yet annotated from the filtered set
        available_indices = [i for i in filtered_df.index if i not in self.annotations]
        
        if len(available_indices) == 0:
            with self.annotation_output: print("No more cells to annotate!")
            return
            
        # Sample
        count = min(n, len(available_indices))
        new_indices = random.sample(available_indices, count)
        
        # Pre-generate tiles
        with self.annotation_output:
            clear_output(wait=True)
            print(f"Pre-generating {len(new_indices)} tiles... This may take a moment.")
            
            for i, idx in enumerate(new_indices):
                try:
                    self.progress_label.value = f"Generating tile {i+1}/{len(new_indices)}"
                    row = self.df.loc[idx]
                    tile = extract_multichannel_tile(row, self.config)
                    self.cached_tiles[idx] = tile
                except Exception as e:
                    print(f"Error extracting tile {idx}: {e}")
                    # Remove bad index from list
                    new_indices[i] = None 
            
            new_indices = [x for x in new_indices if x is not None]
            
        # Add to queue
        self.annotation_queue.extend(new_indices)
        
        # If not currently annotating, start
        if self.current_annotation_index == -1 and self.annotation_queue:
            self._next_annotation()
        else:
            self._update_progress()
            
    def reset_annotations(self, btn):
        self.annotations = {}
        self.annotation_queue = []
        self.current_annotation_index = -1
        self.cached_tiles = {}
        with self.annotation_output: clear_output()
        self._update_progress()
        
    def _next_annotation(self):
        if not self.annotation_queue:
            self.current_annotation_index = -1
            with self.annotation_output:
                clear_output()
                print("Queue empty. Add more samples or Train.")
            return

        self.current_annotation_index = self.annotation_queue.pop(0)
        self._show_current_tile()
        self._update_progress()
        
    def _show_current_tile(self):
        idx = self.current_annotation_index
        row = self.df.loc[idx]
        
        with self.annotation_output:
            clear_output(wait=True)
            try:
                # Use cached tile if available, else extract
                if idx in self.cached_tiles:
                    tile = self.cached_tiles[idx]
                else:
                    # Fallback
                    tile = extract_multichannel_tile(row, self.config)
                    self.cached_tiles[idx] = tile

                # Render with current mappings
                mappings = self.channel_widget.get_mappings()
                rgb = create_rgb_composite(tile, mappings)
                
                # Plot
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(rgb)
                ax.set_title(f"Cell Index: {idx}")
                ax.axis('off')
                plt.show()
                
            except Exception as e:
                print(f"Error loading tile for {idx}: {e}")
                
    def annotate_current(self, label):
        if self.current_annotation_index != -1:
            self.annotations[self.current_annotation_index] = label
            self._next_annotation()
            
    def skip_current(self, btn):
        self._next_annotation()
        
    def _handle_keypress(self, change):
        val = change['new']
        if not val:
            return  # Empty string (reset)
            
        # Immediately reset the text box so it's ready for the next key
        self.keyboard_listener.value = ''
        
        val = val[-1].lower() # Just grab the last typed character in case they type fast
        
        if val == 's':
            self.skip_current(None)
        else:
            # Check numbers 1..N
            for i, cls in enumerate(self.class_names):
                if val == str(i + 1):
                    self.annotate_current(cls)
                    break
        
    def _update_progress(self):
        total = len(self.annotations)
        queue = len(self.annotation_queue)
        self.progress_label.value = f"Annotated: {total} | In Queue: {queue}"

    # --- TRAINING LOGIC ---

    def _get_feature_columns(self):
        # Same exclusion logic as before
        exclude_prefixes = ['Metadata_', 'Image_', 'Location_', 'FileName_', 'PathName_']
        exclude_cols = ['label', 'filename', 'path', 'well', 'field', 'object_number', 'image_number', 
                        'umap_x', 'umap_y', 'UMAP1', 'UMAP2', 'leiden', 'kmeans', 'predicted_label', 'prediction_confidence']
        
        cols = [c for c in self.df.columns if c not in exclude_cols]
        for prefix in exclude_prefixes:
            cols = [c for c in cols if not c.startswith(prefix)]
            
        return self.df[cols].select_dtypes(include=[np.number]).columns.tolist()

    def train_model(self, btn):
        if len(self.annotations) < 10:
            self.log_train("Error: Annotate at least 10 cells first.")
            return
            
        if XGBClassifier is None:
             self.log_train("Error: XGBoost not installed.")
             return
             
        with self.train_log_output:
            clear_output()
            print(f"Training on {len(self.annotations)} samples...")
            
            # Prepare Data
            indices = list(self.annotations.keys())
            labels = list(self.annotations.values())
            
            X = self.df.loc[indices, self._get_feature_columns()]
            
            self.le = LabelEncoder()
            y = self.le.fit_transform(labels)
            
            # Helper to check class balance
            from collections import Counter
            counts = Counter(labels)
            print(f"Class distribution: {counts}")
            
            if len(counts) < 2:
                print("Error: Need at least 2 classes to train.")
                return

            # Train/Test Split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size_slider.value, random_state=42, stratify=y
                )
            except ValueError:
                print("Warning: Not enough samples for stratified split. Using random split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size_slider.value, random_state=42
                )
            
            print("Fitting XGBoost...")
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                eval_metric='logloss',
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            print("Evaluating...")
            self.y_pred = self.model.predict(X_test)
            self.y_test = y_test
            
            self.show_results()
            print("Training Complete.")

    def show_results(self):
        with self.result_output:
            clear_output()
            
            classes = self.le.classes_
            print("--- Classification Report ---")
            print(classification_report(self.y_test, self.y_pred, target_names=classes))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            self.ax_metrics.clear()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.ax_metrics,
                        xticklabels=classes, yticklabels=classes)
            self.ax_metrics.set_ylabel('True')
            self.ax_metrics.set_xlabel('Predicted')
            self.ax_metrics.set_title('Confusion Matrix')
            
            self.fig_metrics.tight_layout()
            
            import matplotlib
            backend = matplotlib.get_backend().lower()
            if 'ipympl' in backend or 'widget' in backend:
                 display(self.fig_metrics.canvas)
            else:
                 display(self.fig_metrics)

    def save_trained_model(self, btn):
        if self.model is None:
            self.log_train("Error: Train a model first.")
            return
            
        filename = self.save_model_input.value
        # Default to UBJSON if no extension provided or empty
        if not filename: 
            filename = 'xgb_model.ubj'
        elif '.' not in filename:
            filename += '.ubj'
        
        try:
            self.model.save_model(filename)
            self.log_train(f"Model saved as {filename}")
        except Exception as e:
            self.log_train(f"Error saving model: {e}")

    # --- PREDICTION LOGIC ---

    def predict_all(self, btn):
        if self.model is None:
            self.log_predict("Error: Train a model first.")
            return
            
        with self.predict_log_output:
            print("Predicting on full dataset...")
            X_all = self.df[self._get_feature_columns()]
            
            y_pred_idx = self.model.predict(X_all)
            y_pred_labels = self.le.inverse_transform(y_pred_idx)
            
            # Confidence
            y_prob = self.model.predict_proba(X_all)
            y_conf = np.max(y_prob, axis=1)
            
            # Store in DF
            self.df['Metadata_PredictedClass'] = y_pred_labels
            self.df['Metadata_PredictionConfidence'] = y_conf
            
            # Export to global Jupyter environment
            import __main__
            __main__.classified_df = self.df
            print("Successfully saved dataframe to the environment as `classified_df`!")
            
            print(f"Prediction complete for {len(self.df)} cells.")
            print(self.df['Metadata_PredictedClass'].value_counts())
            
    def save_predictions(self, btn):
        if 'Metadata_PredictedClass' not in self.df.columns:
            self.log_predict("Error: Run prediction first.")
            return
            
        suffix = self.save_name_input.value
        if not suffix: suffix = '_classified.csv'
        
        # Construct filename from original path if possible, or generic
        # Taking a guess at output path or just using current dir
        out_path = Path(f"predictions{suffix}")
        
        try:
            self.df.to_csv(out_path, index=False)
            self.log_predict(f"Saved to {out_path.absolute()}")
        except Exception as e:
            self.log_predict(f"Error saving: {e}")

    def display(self):
        display(self.tabs)
