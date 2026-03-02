import ipywidgets as widgets
from IPython.display import display
import os
import re
import numpy as np
from collections import defaultdict
import random

try:
    from skimage import io, exposure
    from PIL import Image
    from tqdm.notebook import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    pass

class ImageHeatmapUI:
    """
    UI for generating pseudo-color heatmaps from single microscope channels.
    Supports CV8000, CV8000_Stitched, and CQ1 naming conventions.
    """
    
    PATTERNS = {
        'CV8000': {
            'regex': r"_([A-Z]\d{2})_T.*?F(\d+).*?C(\d+)",
            'desc': 'Example: MMC0015_B07_T0001F001L01A01Z01C01.tiff'
        },
        'CV8000_Stitched': {
            'regex': r"([A-Z]-\d{2})_F(\d+).*?_C(\d+)",
            'desc': 'Example: B-02_F0001_T0001_Z0001_C01.tiff'
        },
        'CQ1': {
            'regex': r"(W\d+)F(\d+).*?C(\d+)",
            'desc': 'Example: W0014F0001T0001Z000C1.tiff'
        }
    }
    
    # User requested Magma, FireIce.
    # Matplotlib standard: magma, inferno, plasma, viridis, cividis, hot, coolwarm, bwr
    COLORMAPS = ['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'hot', 'coolwarm', 'bwr', 'seismic']

    def __init__(self):
        self.detected_channels = []
        
        self._create_ui()
        
    def _create_ui(self):
        # 1. IO Configuration
        self.dd_type = widgets.Dropdown(options=list(self.PATTERNS.keys()), description='Type:', value='CV8000')
        self.dd_type.observe(self._on_type_change, names='value')
        self.lbl_pattern_desc = widgets.Label(value=self.PATTERNS['CV8000']['desc'])
        
        self.txt_input = widgets.Text(placeholder=r"Y:\Data\Images", description='Input:')
        self.txt_output = widgets.Text(placeholder=r"Y:\Data\Heatmaps", description='Output:')
        
        self.btn_scan = widgets.Button(description="Detect Channels", button_style='info')
        self.btn_scan.on_click(self._scan_channels)
        
        io_box = widgets.VBox([
            widgets.HBox([self.dd_type, self.lbl_pattern_desc]),
            self.txt_input,
            self.txt_output,
            self.btn_scan
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='10px 0'))
        
        # 2. Heatmap Settings
        self.dd_target_channel = widgets.Dropdown(description='Channel:', options=[], disabled=True)
        self.dd_colormap = widgets.Dropdown(description='Colormap:', options=self.COLORMAPS, value='magma')
        
        config_box = widgets.VBox([
            widgets.HTML("<b>Heatmap Configuration</b>"),
            self.dd_target_channel,
            self.dd_colormap
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='10px 0'))
        
        # 3. Advanced Settings
        self.chk_subset = widgets.Checkbox(value=True, description='Use Subset for Scaling')
        self.int_subset = widgets.IntText(value=20, description='Size:', layout=widgets.Layout(width='150px'))
        self.flt_low = widgets.FloatText(value=2.0, description='Low %:', step=0.1, layout=widgets.Layout(width='150px'))
        self.flt_high = widgets.FloatText(value=99.8, description='High %:', step=0.1, layout=widgets.Layout(width='150px'))
        self.chk_parallel = widgets.Checkbox(value=False, description='Use Parallel Processing')
        self.int_workers = widgets.IntText(value=4, description='Workers:', layout=widgets.Layout(width='120px'))
        
        settings_box = widgets.VBox([
            widgets.HTML("<b>Advanced Settings</b>"),
            widgets.HBox([self.chk_subset, self.int_subset]),
            widgets.HBox([self.flt_low, self.flt_high]),
            widgets.HBox([self.chk_parallel, self.int_workers])
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='10px 0'))
        
        # 4. Actions
        self.btn_preview = widgets.Button(description="Run Preview (3 Random)", button_style='warning', layout=widgets.Layout(width='48%'))
        self.btn_preview.on_click(self._run_preview)
        
        self.btn_run = widgets.Button(description="Run Heatmap Gen", button_style='success', layout=widgets.Layout(width='48%'))
        self.btn_run.on_click(self._run_gen)
        
        self.out_preview = widgets.Output(layout={'border': '1px solid #eee', 'min_height': '100px'})
        self.out_log = widgets.Output(layout={'height': '200px', 'overflow_y': 'scroll', 'border': '1px solid black'})
        
        self.main_layout = widgets.VBox([
            widgets.HTML("<h2>Pixel Intensity Heatmap Generator</h2>"),
            io_box,
            config_box,
            settings_box,
            widgets.HBox([self.btn_preview, self.btn_run]),
            widgets.HTML("<b>Preview Results:</b>"),
            self.out_preview,
            widgets.HTML("<b>Log:</b>"),
            self.out_log
        ])
        
    def display(self):
        display(self.main_layout)
        
    def _on_type_change(self, change):
        val = change['new']
        self.lbl_pattern_desc.value = self.PATTERNS[val]['desc']
        
    def _scan_channels(self, _):
        input_dir = self.txt_input.value.strip()
        if not os.path.isdir(input_dir):
            with self.out_log: print(f"❌ Input directory not found: {input_dir}")
            return
            
        ptype = self.dd_type.value
        regex = self.PATTERNS[ptype]['regex']
        pattern = re.compile(regex)
        
        found_channels = set()
        
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]
        if not files:
             with self.out_log: print("❌ No TIFF files found.")
             return
             
        for f in files[:100]:
            match = pattern.search(f)
            if match:
                raw_c = match.group(3)
                norm_c = "C" + str(int(raw_c))
                found_channels.add(norm_c)
        
        if not found_channels:
            with self.out_log: print("❌ No channels detected.")
            return
            
        self.detected_channels = sorted(list(found_channels))
        self.dd_target_channel.options = self.detected_channels
        self.dd_target_channel.disabled = False
        with self.out_log: print(f"✅ Detected Channels: {self.detected_channels}")

    def _get_common_params(self):
        if not self.detected_channels:
            with self.out_log: print("⚠️ Please run detection first.")
            return None
        
        input_dir = self.txt_input.value.strip()
        target_ch = self.dd_target_channel.value
        if not target_ch:
            with self.out_log: print("⚠️ Using first channel as default.")
            target_ch = self.detected_channels[0]
            
        ptype = self.dd_type.value
        regex = self.PATTERNS[ptype]['regex']
        
        return {
            'input_dir': input_dir,
            'target_ch': target_ch,
            'colormap': self.dd_colormap.value,
            'regex': regex,
            'p_low': self.flt_low.value,
            'p_high': self.flt_high.value,
            'sub_size': self.int_subset.value,
            'use_sub': self.chk_subset.value
        }

    def _run_preview(self, _):
        params = self._get_common_params()
        if not params: return
        
        with self.out_log: print("🔎 Generating Preview...")
        self.out_preview.clear_output()
        
        try:
            # 1. Scan Groups (Single Channel only)
            files = self._scan_target_files(params['input_dir'], params['regex'], params['target_ch'])
            if not files:
                with self.out_log: print("❌ No files found for target channel.")
                return

            # 2. Stats
            stats = self._calculate_stats(files, params['p_low'], params['p_high'], 
                                          params['use_sub'], params['sub_size'])
            
            # 3. Sample
            keys = list(files.keys())
            sample_keys = random.sample(keys, min(3, len(keys)))
            
            # 4. Render
            images = []
            for k in sample_keys:
                img = render_heatmap_worker(files[k], stats, params['colormap'])
                if img: images.append((k, img))
            
            # 5. Display
            with self.out_preview:
                display_box = []
                for k, img in images:
                    img.thumbnail((300, 300))
                    import io as pyio
                    b = pyio.BytesIO()
                    img.save(b, format='JPEG')
                    wid_img = widgets.Image(value=b.getvalue(), format='jpg', width=300, height=300)
                    display_box.append(widgets.VBox([widgets.Label(k), wid_img]))
                display(widgets.HBox(display_box))
            
            with self.out_log: print("✅ Preview Ready.")
            
        except Exception as e:
            with self.out_log: print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    def _run_gen(self, _):
        params = self._get_common_params()
        if not params: return
        
        output_dir = self.txt_output.value.strip()
        if not output_dir:
            output_dir = os.path.join(params['input_dir'], "Heatmaps_" + params['target_ch'])
            self.txt_output.value = output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        use_parallel = self.chk_parallel.value
        workers = self.int_workers.value if use_parallel else 1
        
        with self.out_log:
            print(f"🚀 Starting Heatmap Gen -> {output_dir}")
            
        try:
            files = self._scan_target_files(params['input_dir'], params['regex'], params['target_ch'])
            if not files: return
            
            stats = self._calculate_stats(files, params['p_low'], params['p_high'], 
                                          params['use_sub'], params['sub_size'])
            
            sorted_items = sorted(files.items())
            count = 0
            
            if use_parallel and workers > 1:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    for k, path in sorted_items:
                        out_path = os.path.join(output_dir, f"{k}_heatmap.jpg")
                        fut = executor.submit(render_save_heatmap, path, stats, params['colormap'], out_path)
                        futures[fut] = k
                    
                    for f in tqdm(as_completed(futures), total=len(files), desc="Parallel Heatmap"):
                        if f.result(): count += 1
            else:
                for k, path in tqdm(sorted_items, desc="Heatmaps"):
                    img = render_heatmap_worker(path, stats, params['colormap'])
                    if img:
                        out_path = os.path.join(output_dir, f"{k}_heatmap.jpg")
                        img.save(out_path, quality=90)
                        count += 1
                        
            with self.out_log: print(f"✅ Saved {count} heatmaps.")
            
        except Exception as e:
            with self.out_log: print(f"❌ Error: {e}")

    # Helpers
    def _scan_target_files(self, input_dir, regex_str, target_ch):
        pattern = re.compile(regex_str)
        target_files = {} # Key: Well_Field, Value: Path
        
        valid_exts = ('.tif', '.tiff', '.png', '.jpg')
        for filename in os.listdir(input_dir):
             if filename.lower().endswith(valid_exts):
                 match = pattern.search(filename)
                 if match:
                     well_id, field_id, channel_key = match.groups()
                     norm_c = "C" + str(int(channel_key))
                     if norm_c == target_ch:
                         group_key = f"{well_id}_{field_id}"
                         target_files[group_key] = os.path.join(input_dir, filename)
        return target_files

    def _calculate_stats(self, target_files, p_low, p_high, use_subset, subset_size):
        pixels = []
        keys = list(target_files.keys())
        
        if use_subset and len(keys) > subset_size:
            sample_keys = random.sample(keys, subset_size)
        else:
            sample_keys = keys
            
        for k in sample_keys:
            path = target_files[k]
            img = io.imread(path)
            pixels.extend(np.random.choice(img.ravel(), size=min(img.size, 10000)))
            
        if not pixels: return (0, 255)
        return np.percentile(pixels, (p_low, p_high))

# Top-level for pickling
def render_heatmap_worker(path, stats, colormap_name):
    img = io.imread(path)
    v_min, v_max = stats
    if v_max <= v_min: v_max = v_min + 1e-5
    
    # Normalize 0-1
    norm = (img.astype(np.float32) - v_min) / (v_max - v_min)
    norm = np.clip(norm, 0, 1)
    
    # Apply Colormap
    cmap = plt.get_cmap(colormap_name)
    rgba = cmap(norm) # Returns (H, W, 4) floats 0-1
    
    # Convert to RGB uint8
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)

def render_save_heatmap(path, stats, colormap_name, out_path):
    try:
        img = render_heatmap_worker(path, stats, colormap_name)
        if img:
            img.save(out_path, quality=90)
            return True
        return False
    except Exception as e:
        print(f"Error {path}: {e}")
        return False
