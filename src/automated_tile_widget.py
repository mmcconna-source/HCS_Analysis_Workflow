import os
import datetime
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    from src.tile_extraction import (
        NotebookConfig, ChannelMapping, extract_multichannel_tile, 
        create_rgb_composite, export_tiles, create_default_mappings, resolve_image_paths
    )
except ImportError:
    try:
        from tile_extraction import (
            NotebookConfig, ChannelMapping, extract_multichannel_tile, 
            create_rgb_composite, export_tiles, create_default_mappings, resolve_image_paths
        )
    except ImportError:
        logging.warning("Could not import tile_extraction. Tile generation features may be limited.")

try:
    from src.channel_mapping_widget import ChannelMappingWidget
except ImportError:
    try:
        from channel_mapping_widget import ChannelMappingWidget
    except ImportError:
        ChannelMappingWidget = None
        logging.warning("Could not import ChannelMappingWidget. Interactive tile config disabled.")

logger = logging.getLogger(__name__)


class AutomatedTileExportWidget:
    """
    Widget to load analytical data (UMAP/Leiden clusters), display a sample 
    of 10 images to tune channel mappings, and automate exporting N tiles 
    per cluster into structured sub-directories.
    """
    
    def __init__(self, df: pd.DataFrame, config: NotebookConfig, output_root: str = "Automated_Tiles"):
        self.df = df.copy().reset_index(drop=True)
        self.config = config
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.cm_widget = None
        
        self.output_area = widgets.Output()
        self.preview_area = widgets.Output()
        
        self._create_widgets()
        
    def _create_widgets(self):
        # 1. Loading/Saving Elements
        self.save_name_input = widgets.Text(description='Folder Name:', value=f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
        self.load_data_btn = widgets.Button(description='Load DF (Pickle/CSV)', button_style='info')
        self.load_data_btn.on_click(self.load_df_data)
        
        self.metadata_dropdown = widgets.Dropdown(description='By Cluster:')
        
        # 2. Preview UI
        self.generate_preview_btn = widgets.Button(description='Load 10 Random Cells', button_style='warning', layout=widgets.Layout(width='auto'))
        self.generate_preview_btn.on_click(self.generate_previews)

        # 3. Export UI
        self.export_name_input = widgets.Text(description='Export Name:', value='exported_tiles', tooltip='Name for the root output folder')
        self.tiles_per_cluster_input = widgets.IntText(value=50, description='Tiles/Cluster:', layout=widgets.Layout(width='200px'))
        
        self.automated_export_btn = widgets.Button(description='Batch Export Tiles', button_style='success')
        self.automated_export_btn.on_click(self.run_batch_export)

        self._update_color_options()
        self._build_layout()

    def _build_layout(self):
        self.loading_box = widgets.VBox([
            widgets.HTML("<h3>1. Data Sourcing</h3>"),
            widgets.HTML("<i>Enter the 'Folder Name' you used in the UMAP Widget and click Load.</i>"),
            self.save_name_input,
            self.load_data_btn,
            widgets.HTML("<hr>")
        ])
        
        self.preview_box = widgets.VBox([
            widgets.HTML("<h3>2. Color Configuration</h3>"),
            widgets.HTML("<i>Generate random previews and use the sliders to configure brightness/contrast.</i>"),
            self.generate_preview_btn,
            self.preview_area,
            widgets.HTML("<hr>")
        ])
        
        self.export_box = widgets.VBox([
            widgets.HTML("<h3>3. Automated Batch Export</h3>"),
            widgets.HTML("<i>Select a clustering column (like 'leiden'), choose how many tiles per group you want, and click Export!</i>"),
            self.metadata_dropdown,
            self.export_name_input,
            self.tiles_per_cluster_input,
            self.automated_export_btn
        ])
        
        self.main_layout = widgets.VBox([
            self.loading_box, self.preview_box, self.export_box, self.output_area
        ])

    def display(self):
        display(self.main_layout)

    def _update_color_options(self):
        """Update options for grouping (leiden, kmeans, metadata)."""
        meta_options = [c for c in self.df.columns if c.startswith('Metadata_') or c in ['leiden', 'kmeans']]
        self.metadata_dropdown.options = ['None'] + sorted(meta_options)
        
        if 'leiden' in meta_options:
            self.metadata_dropdown.value = 'leiden'

    def load_df_data(self, btn):
        name = self.save_name_input.value
        # For compatibility with UMAP exploration widget defaults
        # Often UMAP creates a folder "UMAP" so we should search relative to that if present.
        # Allow checking a parent 'UMAP' folder if local isn't found
        target_path_local = self.output_root / name
        target_path_umap = Path("UMAP") / name
        
        load_dir = target_path_local
        if not load_dir.exists():
            if target_path_umap.exists():
                load_dir = target_path_umap
            else:
                 with self.output_area: print(f"Could not locate folder {name} in {self.output_root} or 'UMAP/'.")
                 return
            
        csv_path = load_dir / "umap_data.csv"
        pkl_path = load_dir / "umap_data.pkl"
        
        try:
            if pkl_path.exists():
                with self.output_area: print(f"Loading {pkl_path}...")
                self.df = pd.read_pickle(pkl_path)
            elif csv_path.exists():
                with self.output_area: print(f"Loading {csv_path}...")
                self.df = pd.read_csv(csv_path)
            else:
                with self.output_area: print(f"No umap_data.pkl or umap_data.csv found in {load_dir}.")
                return
                 
            self._update_color_options()
            with self.output_area: print(f"Loaded {len(self.df)} cells successfully!")
        except Exception as e:
            with self.output_area: print(f"Error loading data: {e}")

    def generate_previews(self, btn):
        if len(self.df) == 0:
            with self.output_area: print("DataFrame is empty. Load data first.")
            return
            
        with self.output_area: print("Sampling 10 random cells to preview... (Please wait)")
        
        sample_size = min(10, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=np.random.randint(0, 10000))
        
        preview_tiles = []
        for _, row in sample_df.iterrows():
            try:
                tile = extract_multichannel_tile(row, self.config, padding_mode='constant')
                preview_tiles.append(tile)
            except Exception as e:
                # If image missing or error, skip
                continue
                
        if not preview_tiles:
            with self.output_area: print("Failed to extract any tiles for preview. Check configured paths.")
            return
            
        with self.preview_area:
            clear_output()
            # Feed lists into CM widget so it handles merging automatically
            self.cm_widget = ChannelMappingWidget(self.config, sample_tiles=preview_tiles)
            self.cm_widget.display()
            
        with self.output_area: print("Previews generated!")

    def run_batch_export(self, btn):
        group_col = self.metadata_dropdown.value
        if group_col == 'None' or group_col not in self.df.columns:
            with self.output_area: print("Please select a valid cluster/grouping column.")
            return
            
        tiles_per_cluster = self.tiles_per_cluster_input.value
        export_name = self.export_name_input.value
        
        if not export_name:
             with self.output_area: print("Please enter an export name.")
             return
             
        mappings = None
        if self.cm_widget is not None:
             mappings = self.cm_widget.get_mappings()
        else:
             with self.output_area: print("WARNING: No previews generated. Using default/blank channel mappings.")
             
        unique_groups = sorted(list(self.df[group_col].dropna().unique()))
        
        # Create Root Directory
        root_tiles_dir = self.output_root / export_name
        root_tiles_dir.mkdir(parents=True, exist_ok=True)
        
        with self.output_area:
            print(f"Starting Automated Batch Export: {tiles_per_cluster} tiles per cluster...")
            print(f"Targeting groups in '{group_col}': {unique_groups}")
        
        total_success = 0
        total_fails = 0
        
        import random
        for group in unique_groups:
            group_df = self.df[self.df[group_col] == group]
            max_tiles = min(tiles_per_cluster, len(group_df))
            
            # Subsample
            sample_indices = random.sample(list(group_df.index), max_tiles)
            subset = self.df.loc[sample_indices]
            
            # Prepare cluster subdirectory
            cluster_dir = root_tiles_dir / str(group)
            cluster_dir.mkdir(parents=True, exist_ok=True)
            
            with self.output_area: print(f"  - Exporting {max_tiles} tiles for Cluster {group}...")
            
            # Clone config and update the output directory specifically for this loop
            batch_config = NotebookConfig(
                csv_path=self.config.csv_path,
                image_base_path=self.config.image_base_path,
                output_dir=str(cluster_dir),
                tile_size=self.config.tile_size,
                x_column=self.config.x_column,
                y_column=self.config.y_column,
                umap_x_column=self.config.umap_x_column,
                umap_y_column=self.config.umap_y_column,
                well_column=self.config.well_column,
                field_column=self.config.field_column,
                channel_names=self.config.channel_names,
                filename_pattern=self.config.filename_pattern,
                bbox_min_x_column=self.config.bbox_min_x_column,
                bbox_min_y_column=self.config.bbox_min_y_column,
                bbox_max_x_column=self.config.bbox_max_x_column,
                bbox_max_y_column=self.config.bbox_max_y_column
            )
            
            try:
                res = export_tiles(subset, batch_config, mappings, output_format='png')
                total_success += res['successful']
                total_fails += res['failed']
                if res['failed'] > 0:
                    with self.output_area:
                        print(f"    ! Failed to export {res['failed']} tiles for Cluster {group}.")
            except Exception as e:
                with self.output_area:
                    print(f"    ! Export completely failed for Cluster {group}: {e}")
                    
        with self.output_area: 
            print("====================================")
            print(f"Batch Export Finished!")
            print(f"Successfully generated {total_success} tiles across all clusters.")
            if total_fails > 0:
                print(f"Failed extracting {total_fails} tiles.")
            print(f"Files saved to: {root_tiles_dir.resolve()}")
