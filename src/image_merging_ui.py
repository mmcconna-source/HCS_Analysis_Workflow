import ipywidgets as widgets
from IPython.display import display
import os
import re
import numpy as np
from collections import defaultdict
import random
# Try-catch imports for heavy libs in case they are missing in dev env, 
# but User has them.
try:
    from skimage import io, exposure
    from PIL import Image
    from tqdm.notebook import tqdm
except ImportError:
    pass

class ImageMergerUI:
    """
    UI for merging grayscale TIFF channels into RGB JPEGs.
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
    
    COLORS = ['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow', 'Gray']
    COLOR_WEIGHTS = {
        'RED': (1,0,0), 'GREEN': (0,1,0), 'BLUE': (0,0,1), 
        'CYAN': (0,1,1), 'MAGENTA': (1,0,1), 'YELLOW': (1,1,0), 
        'GRAY': (1,1,1)
    }

    def __init__(self):
        self.detected_channels = []
        self.channel_configs = {} # {channel_id: (color_widget, intensity_widget)}
        
        self._create_ui()
        
        # Initial scan/update based on default
        # self._on_type_change(None) # Can't do this yet without a dir, so valid check helps
        
    def _create_ui(self):
        # 1. IO Configuration
        self.dd_type = widgets.Dropdown(
            options=list(self.PATTERNS.keys()),
            description='Type:',
            value='CV8000'
        )
        self.dd_type.observe(self._on_type_change, names='value')
        
        self.lbl_pattern_desc = widgets.Label(value=self.PATTERNS['CV8000']['desc'])
        
        self.txt_input = widgets.Text(placeholder=r"Y:\Data\Images", description='Input:')
        self.txt_output = widgets.Text(placeholder=r"Y:\Data\Merged", description='Output:')
        
        self.btn_scan = widgets.Button(description="Detect Channels", button_style='info')
        self.btn_scan.on_click(self._scan_channels)
        
        io_box = widgets.VBox([
            widgets.HBox([self.dd_type, self.lbl_pattern_desc]),
            self.txt_input,
            self.txt_output,
            self.btn_scan
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='10px 0'))
        
        # 2. Channel Configuration
        self.channel_box = widgets.VBox([])
        self.channel_container = widgets.VBox([
            widgets.HTML("<b>Channel Configuration</b> (Run Detection first)"),
            self.channel_box
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
        
        self.btn_run = widgets.Button(description="Run Full Merge", button_style='success', layout=widgets.Layout(width='48%'))
        self.btn_run.on_click(self._run_merge)
        
        self.out_preview = widgets.Output(layout={'border': '1px solid #eee', 'min_height': '100px'})
        self.out_log = widgets.Output(layout={'height': '200px', 'overflow_y': 'scroll', 'border': '1px solid black'})
        
        self.main_layout = widgets.VBox([
            widgets.HTML("<h2>JPEG Channel Merger</h2>"),
            io_box,
            self.channel_container,
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
             with self.out_log: print("❌ No TIFF files found in input directory.")
             return
             
        for f in files[:100]:
            match = pattern.search(f)
            if match:
                raw_c = match.group(3)
                norm_c = "C" + str(int(raw_c))
                found_channels.add(norm_c)
        
        if not found_channels:
            with self.out_log: print("❌ No channels detected. Check regex/naming.")
            return
            
        self.detected_channels = sorted(list(found_channels))
        self._build_channel_ui()
        with self.out_log: print(f"✅ Detected Channels: {self.detected_channels}")

    def _build_channel_ui(self):
        self.channel_configs = {}
        rows = []
        
        for i, ch in enumerate(self.detected_channels):
            # Heuristic assignment
            def_col = self.COLORS[i % len(self.COLORS)]
            
            dd_col = widgets.Dropdown(options=self.COLORS, value=def_col, layout=widgets.Layout(width='100px'))
            flt_int = widgets.FloatText(value=1.0, step=0.1, layout=widgets.Layout(width='60px'))
            
            self.channel_configs[ch] = (dd_col, flt_int)
            
            rows.append(widgets.HBox([
                widgets.Label(f"Channel {ch}:", layout=widgets.Layout(width='80px')),
                dd_col,
                widgets.Label("Intensity:", layout=widgets.Layout(width='60px')),
                flt_int
            ]))
            
        self.channel_box.children = rows

    def _get_common_params(self):
        """Helper to gather UI params."""
        if not self.detected_channels:
            with self.out_log: print("⚠️ Please run detection first.")
            return None
        
        input_dir = self.txt_input.value.strip()
        
        # Build Config Map {'C1': ('BLUE', 1.0)}
        channel_map = {}
        for ch, (dd, flt) in self.channel_configs.items():
            channel_map[ch] = (dd.value.upper(), flt.value)
            
        ptype = self.dd_type.value
        regex = self.PATTERNS[ptype]['regex']
        
        return {
            'input_dir': input_dir,
            'channel_map': channel_map,
            'regex': regex,
            'p_low': self.flt_low.value,
            'p_high': self.flt_high.value,
            'sub_size': self.int_subset.value,
            'use_sub': self.chk_subset.value
        }

    def _run_preview(self, _):
        params = self._get_common_params()
        if not params: return
        
        with self.out_log: print("🔎 Generating Preview (Random 3)...")
        self.out_preview.clear_output()
        
        try:
            # 1. Scan Groups
            groups = self._scan_groups(params['input_dir'], params['regex'], params['channel_map'])
            if not groups: 
                with self.out_log: print("❌ No image groups found.")
                return
            
            # 2. Calculate Stats
            stats = self._calculate_stats(groups, params['channel_map'], params['input_dir'],
                                          params['p_low'], params['p_high'], 
                                          params['use_sub'], params['sub_size'])
            
            # 3. Sample
            keys = list(groups.keys())
            sample_keys = random.sample(keys, min(3, len(keys)))
            
            # 4. Render
            images = []
            for k in sample_keys:
                img = self._render_group(groups[k], params['channel_map'], stats)
                if img: images.append((k, img))
            
            # 5. Display
            with self.out_preview:
                display_box = []
                for k, img in images:
                    # Convert to fit safely in UI (max width 300?)
                    # If huge, thumbnail it
                    img.thumbnail((300, 300))
                    
                    # Widget Image requires bytes. 
                    # Easiest is just display(img) direct in loop if we use output context
                    # But let's put them in an HBox
                    
                    # We can use PIL to widgets.Image
                    import io as pyio
                    b = pyio.BytesIO()
                    img.save(b, format='JPEG')
                    
                    wid_img = widgets.Image(value=b.getvalue(), format='jpg', width=300, height=300)
                    display_box.append(widgets.VBox([
                        widgets.Label(f"Sample: {k}"),
                        wid_img
                    ]))
                
                display(widgets.HBox(display_box))
                
            with self.out_log: print("✅ Preview Ready.")
            
        except Exception as e:
            with self.out_log: print(f"❌ Preview Error: {e}")
            import traceback
            traceback.print_exc()

    def _run_merge(self, _):
        params = self._get_common_params()
        if not params: return
        
        output_dir = self.txt_output.value.strip()
        if not output_dir:
            output_dir = os.path.join(params['input_dir'], "Merged_JPEGs")
            self.txt_output.value = output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        with self.out_log:
            print(f"🚀 Starting Full Merge -> {output_dir}")
            print(f"   Groups will be saved as JPEGs...")
            
        try:
            groups = self._scan_groups(params['input_dir'], params['regex'], params['channel_map'])
            if not groups: return
            
            stats = self._calculate_stats(groups, params['channel_map'], params['input_dir'],
                                          params['p_low'], params['p_high'], 
                                          params['use_sub'], params['sub_size'])
            
            count = 0
            for k, channels in tqdm(sorted(groups.items()), desc="Merging"):
                img = self._render_group(channels, params['channel_map'], stats)
                if img:
                    out_name = os.path.join(output_dir, f"{k}_merged.jpg")
                    img.save(out_name, quality=90)
                    count += 1
            
            with self.out_log: print(f"✅ Full Merge Complete. Saved {count} images.")
            
        except Exception as e:
            with self.out_log: print(f"❌ Error: {e}")

    # --- Processing Helpers ---

    def _scan_groups(self, input_dir, regex_str, channel_map):
        pattern = re.compile(regex_str)
        image_groups = defaultdict(dict)
        valid_exts = ('.tif', '.tiff', '.png', '.jpg')
        
        # Scandir is faster? os.listdir is fine for now.
        for filename in os.listdir(input_dir):
             if filename.lower().endswith(valid_exts):
                 match = pattern.search(filename)
                 if match:
                     well_id, field_id, channel_key = match.groups()
                     norm_c = "C" + str(int(channel_key))
                     if norm_c in channel_map:
                         group_key = f"{well_id}_{field_id}"
                         image_groups[group_key][norm_c] = os.path.join(input_dir, filename)
        return image_groups

    def _calculate_stats(self, image_groups, channel_map, input_dir, p_low, p_high, use_subset, subset_size):
        global_stats = {ch: [] for ch in channel_map.keys()}
        group_keys = list(image_groups.keys())
        
        if use_subset and len(group_keys) > subset_size:
            sample_keys = random.sample(group_keys, subset_size)
        else:
            sample_keys = group_keys
            
        for g_key in sample_keys:
            for ch_id in channel_map.keys():
                 if ch_id in image_groups[g_key]:
                     path = image_groups[g_key][ch_id]
                     img = io.imread(path)
                     global_stats[ch_id].extend(np.random.choice(img.ravel(), size=min(img.size, 10000)))
                     
        scaling_registry = {}
        for ch_id, pixels in global_stats.items():
            if pixels:
                v_min, v_max = np.percentile(pixels, (p_low, p_high))
                scaling_registry[ch_id] = (v_min, v_max)
        
        return scaling_registry

    def _render_group(self, channels, channel_map, scaling_registry):
        # Wrapper for preview usage (keeps it simple within class for single-thread preview)
        return render_worker(channels, channel_map, scaling_registry, self.COLOR_WEIGHTS)
        
    def _run_merge(self, _):
        params = self._get_common_params()
        if not params: return
        
        output_dir = self.txt_output.value.strip()
        if not output_dir:
            output_dir = os.path.join(params['input_dir'], "Merged_JPEGs")
            self.txt_output.value = output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        use_parallel = self.chk_parallel.value
        workers = self.int_workers.value if use_parallel else 1
        
        with self.out_log:
            print(f"🚀 Starting Full Merge -> {output_dir}")
            print(f"   Mode: {'Parallel (' + str(workers) + ' workers)' if use_parallel else 'Sequential'}")
            
        try:
            groups = self._scan_groups(params['input_dir'], params['regex'], params['channel_map'])
            if not groups: return
            
            stats = self._calculate_stats(groups, params['channel_map'], params['input_dir'],
                                          params['p_low'], params['p_high'], 
                                          params['use_sub'], params['sub_size'])
            
            sorted_items = sorted(groups.items())
            total = len(sorted_items)
            count = 0
            
            if use_parallel and workers > 1:
                # Parallel Execution
                from concurrent.futures import ProcessPoolExecutor, as_completed
                
                # Prepare args
                tasks = []
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    for k, channels in sorted_items:
                        out_path = os.path.join(output_dir, f"{k}_merged.jpg")
                        # Submit task (render + save)
                        # We need a new wrapper that does SAVE as well to avoid passing images back
                        fut = executor.submit(render_and_save_worker, channels, params['channel_map'], 
                                              stats, self.COLOR_WEIGHTS, out_path)
                        futures[fut] = k
                        
                    for f in tqdm(as_completed(futures), total=total, desc="Parallel Merge"):
                        if f.result(): # result is success bool
                            count += 1
            else:
                # Sequential Execution
                for k, channels in tqdm(sorted_items, desc="Sequential Merge"):
                    img = render_worker(channels, params['channel_map'], stats, self.COLOR_WEIGHTS)
                    if img:
                        out_name = os.path.join(output_dir, f"{k}_merged.jpg")
                        img.save(out_name, quality=90)
                        count += 1
            
            with self.out_log: print(f"✅ Full Merge Complete. Saved {count} images.")
            
        except Exception as e:
            with self.out_log: print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

# --- Top Level Functions for Pickling ---

def render_worker(channels, channel_map, scaling_registry, color_weights):
    if not channels: return None
    
    first_path = list(channels.values())[0]
    sample = io.imread(first_path)
    h, w = sample.shape
    composite_f = np.zeros((h, w, 3), dtype=np.float32)
    has_data = False
    
    for ch_id, (col_name, intensity) in channel_map.items():
        if ch_id in channels:
            has_data = True
            img = io.imread(channels[ch_id])
            
            v_min, v_max = scaling_registry.get(ch_id, (img.min(), img.max()))
            if v_max <= v_min: v_max = v_min + 1e-5
            
            rescaled = exposure.rescale_intensity(
                img, in_range=(v_min, v_max), out_range=(0.0, 1.0)
            ).astype(np.float32)
            
            w_rgb = color_weights.get(col_name, (1,1,1))
            for i in range(3):
                if w_rgb[i] > 0:
                    composite_f[:,:,i] += (rescaled * intensity * w_rgb[i])
                    
    if has_data:
        final_rgb = (np.clip(composite_f, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(final_rgb)
    return None

def render_and_save_worker(channels, channel_map, scaling_registry, color_weights, out_path):
    """Worker for parallel execution that handles saving."""
    try:
        img = render_worker(channels, channel_map, scaling_registry, color_weights)
        if img:
            img.save(out_path, quality=90)
            return True
        return False
    except Exception as e:
        print(f"Error processing {out_path}: {e}")
        return False

