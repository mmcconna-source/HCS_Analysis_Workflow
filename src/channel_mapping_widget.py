
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from src.tile_extraction import NotebookConfig, ChannelMapping, create_rgb_composite

class ChannelMappingWidget:
    """Interactive widget for configuring channel-to-RGB mappings."""

    def __init__(self, config: NotebookConfig, sample_tiles: List[np.ndarray] = None):
        self.config = config
        # Ensure sample_tiles is a list
        if sample_tiles is not None and isinstance(sample_tiles, np.ndarray):
             self.sample_tiles = [sample_tiles]
        else:
             self.sample_tiles = sample_tiles
             
        self.mappings: List[ChannelMapping] = []
        self._widgets = {}
        self.widget_container = widgets.VBox([])
        self._preview_output = widgets.Output()
        
        # Stats Cache for Global Scaling
        self._global_stats = {} # {channel_index: (min_val, max_val, abs_max)}
        
        self._init_default_mappings()
        self._build_ui()
        
        if self.sample_tiles:
            self._calculate_global_stats()

    def _calculate_global_stats(self):
        """Calculate 0.1th and 99.9th percentiles across all sample tiles for global scaling."""
        if not self.sample_tiles: return
        
        n_channels = len(self.config.channel_names)
        
        for i in range(n_channels):
            # Aggregate pixels for this channel from all tiles
            pixels = []
            for tile in self.sample_tiles:
                if i < tile.shape[0]:
                    pixels.append(tile[i].flatten())
            
            if pixels:
                all_pixels = np.concatenate(pixels)
                # Robust min/max (ignoring hot pixels/dead pixels)
                p_min = np.percentile(all_pixels, 0.1)
                p_max = np.percentile(all_pixels, 99.9)
                abs_max = np.max(all_pixels)
                
                self._global_stats[i] = (p_min, p_max, abs_max)

    def _init_default_mappings(self):
        default_colors = {
            'DNA': 'B', 'KRT8': 'G', 'CMO': None, 'TP63': None,
            'Phalloidin': 'R', 'DAPI': 'B', 'GFP': 'G', 'RFP': 'R',
            'Cy5': 'R', 'Keratin': 'Y', 'CD45': 'C', 'CD3': 'M'
        }

        for i, name in enumerate(self.config.channel_names):
            self.mappings.append(ChannelMapping(
                channel_index=i,
                channel_name=name,
                target_color=default_colors.get(name, None),
                intensity_min=1.0,
                intensity_max=99.0,
                use_percentile=True,
                gamma=1.0
            ))

    def _build_ui(self):
        channel_widgets = []
        
        color_options = [
            ('None', None), 
            ('Red', 'R'), ('Green', 'G'), ('Blue', 'B'),
            ('Cyan', 'C'), ('Magenta', 'M'), ('Yellow', 'Y')
        ]

        # Global Toggle
        self.global_scale_checkbox = widgets.Checkbox(value=False, description='Use Global Scale (from samples)')
        self.global_scale_checkbox.observe(self._on_global_toggle, names='value')

        for i, mapping in enumerate(self.mappings):
            color_dropdown = widgets.Dropdown(
                options=color_options,
                value=mapping.target_color,
                description=f'{mapping.channel_name}:',
                style={'description_width': '80px'},
                layout=widgets.Layout(width='180px')
            )

            # Sliders (Default to Percentile Mode)
            min_slider = widgets.FloatSlider(
                value=mapping.intensity_min,
                min=0, max=100, step=0.5,
                description='Min %:',
                style={'description_width': '50px'},
                layout=widgets.Layout(width='180px')
            )

            max_slider = widgets.FloatSlider(
                value=mapping.intensity_max,
                min=0, max=100, step=0.5,
                description='Max %:',
                style={'description_width': '50px'},
                layout=widgets.Layout(width='180px')
            )

            gamma_slider = widgets.FloatSlider(
                value=mapping.gamma,
                min=0.1, max=3.0, step=0.1,
                description='Gamma:',
                style={'description_width': '50px'},
                layout=widgets.Layout(width='180px')
            )

            self._widgets[i] = {
                'color': color_dropdown,
                'min': min_slider,
                'max': max_slider,
                'gamma': gamma_slider
            }

            # Observe changes
            color_dropdown.observe(lambda change, idx=i: self._on_change(idx), names='value')
            min_slider.observe(lambda change, idx=i: self._on_change(idx), names='value')
            max_slider.observe(lambda change, idx=i: self._on_change(idx), names='value')
            gamma_slider.observe(lambda change, idx=i: self._on_change(idx), names='value')

            row = widgets.HBox([color_dropdown, min_slider, max_slider, gamma_slider])
            channel_widgets.append(row)

        preview_btn = widgets.Button(description='Update Preview', button_style='info')
        preview_btn.on_click(self._update_preview)

        self.widget_container.children = [
            widgets.HTML('<b>Channel Mapping Configuration</b>'),
            self.global_scale_checkbox,
            widgets.VBox(channel_widgets),
            preview_btn,
            self._preview_output
        ]
        
        if self.sample_tiles:
             self._update_preview(None)

    def _on_global_toggle(self, change):
        is_global = change['new']
        
        for i, mapping in enumerate(self.mappings):
            w = self._widgets[i]
            
            if is_global:
                # Switch to Absolute Mode
                mapping.use_percentile = False
                
                # Get stats
                if i in self._global_stats:
                    p_min, p_max, abs_max = self._global_stats[i]
                else:
                    p_min, p_max, abs_max = 0, 65535, 65535 # Fallback
                
                # Update Sliders to Absolute Range
                # Update Max first to avoid value clipping
                w['min'].max = abs_max * 1.2 
                w['max'].max = abs_max * 1.2
                
                w['min'].description = 'Min Abs:'
                w['max'].description = 'Max Abs:'
                
                w['min'].step = 10
                w['max'].step = 10
                
                # Set values
                w['min'].value = p_min
                w['max'].value = p_max
                
            else:
                # Switch back to Percentile Mode
                mapping.use_percentile = True
                
                w['min'].description = 'Min %:'
                w['max'].description = 'Max %:'
                
                w['min'].max = 100
                w['max'].max = 100
                w['min'].step = 0.5
                w['max'].step = 0.5
                
                # Reset to default percentile values or previous
                w['min'].value = 1.0
                w['max'].value = 99.0
            
            # Trigger value update to mapping object
            self._on_change(i)

    def _on_change(self, idx: int):
        w = self._widgets[idx]
        self.mappings[idx].target_color = w['color'].value
        self.mappings[idx].intensity_min = w['min'].value
        self.mappings[idx].intensity_max = w['max'].value
        self.mappings[idx].gamma = w['gamma'].value
        self.mappings[idx].use_percentile = not self.global_scale_checkbox.value
        # Auto-update preview if lightweight? Maybe better to require click.
        
    def _update_preview(self, btn):
        if not self.sample_tiles:
            return

        with self._preview_output:
            clear_output(wait=True)
            try:
                # Prepare figure with N+1 subplots (N tiles + 1 legend)
                n_samples = len(self.sample_tiles)
                fig, axes = plt.subplots(1, n_samples + 1, figsize=(4 * (n_samples + 1), 4))
                if n_samples == 0: axes = [axes] # Handle edge case
                elif n_samples + 1 == 1: axes = [axes] # Handle edge case
                
                # Render tiles
                for i, tile in enumerate(self.sample_tiles):
                    rgb = create_rgb_composite(tile, self.mappings)
                    axes[i].imshow(rgb)
                    axes[i].set_title(f'Sample {i+1}')
                    axes[i].axis('off')

                # Render Legend
                legend_ax = axes[-1]
                active_mappings = [m for m in self.mappings if m.target_color is not None]
                if active_mappings:
                    info = "\n".join([
                        f"{m.channel_name} -> {m.target_color}" for m in active_mappings
                    ])
                    legend_ax.text(0.1, 0.5, info, va='center', fontsize=12)
                    legend_ax.set_title('Channel Assignments')
                    legend_ax.axis('off')
                else:
                    legend_ax.text(0.5, 0.5, "No channels selected", ha='center')
                    legend_ax.axis('off')

                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Preview error: {e}")
                import traceback
                traceback.print_exc()

    def get_mappings(self) -> List[ChannelMapping]:
        return self.mappings

    def set_sample_tiles(self, tiles: List[np.ndarray]):
        """Update the sample tiles used for preview. Accepts a list of 3D arrays."""
        if isinstance(tiles, np.ndarray):
            tiles = [tiles]
        self.sample_tiles = tiles
        self._update_preview(None)

    def display(self):
        display(self.widget_container)
