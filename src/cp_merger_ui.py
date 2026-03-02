import ipywidgets as widgets
from IPython.display import display
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import time

class CPMergerUI:
    """
    UI for aggregating CellProfiler single-cell CSVs from well-based folders.
    Replicates functionality of Example_Scripts/cp_merger_module.py.
    """
    
    def __init__(self):
        self._create_ui()
        
    def _create_ui(self):
        # 1. IO Config
        self.txt_input = widgets.Text(placeholder="Y:/Data/Analysis_Output", description='Input Root:')
        self.txt_output = widgets.Text(placeholder="Y:/Data/Aggregated", description='Output Dir:')
        
        io_box = widgets.VBox([
            widgets.HTML("<b>Directories</b>"),
            self.txt_input,
            self.txt_output
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='10px 0'))
        
        # 2. File Config
        self.txt_cell = widgets.Text(value="MyExpt_Cell.csv", description='Cell CSV:')
        self.txt_cyto = widgets.Text(value="MyExpt_Cytoplasm.csv", description='Cyto CSV:')
        self.txt_nuc = widgets.Text(value="MyExpt_Nucleus.csv", description='Nuc CSV:')
        self.txt_img = widgets.Text(value="MyExpt_Image.csv", description='Image CSV:')
        
        file_box = widgets.VBox([
            widgets.HTML("<b>Filename Configuration</b>"),
            self.txt_cell,
            self.txt_cyto,
            self.txt_nuc,
            self.txt_img
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='10px 0'))
        
        # 3. Settings
        self.chk_parallel = widgets.Checkbox(value=True, description='Use Parallel Processing')
        self.int_workers = widgets.IntText(value=4, description='Workers:', layout=widgets.Layout(width='120px'))
        
        settings_box = widgets.VBox([
            widgets.HTML("<b>Settings</b>"),
            widgets.HBox([self.chk_parallel, self.int_workers])
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='10px 0'))
        
        # 4. Action
        self.btn_run = widgets.Button(description="Merge Data", button_style='success', layout=widgets.Layout(width='100%'))
        self.btn_run.on_click(self._run_merge)
        
        self.out_log = widgets.Output(layout={'height': '200px', 'overflow_y': 'scroll', 'border': '1px solid black'})
        
        self.main_layout = widgets.VBox([
            widgets.HTML("<h2>CellProfiler Data Merger</h2>"),
            io_box,
            file_box,
            settings_box,
            self.btn_run,
            self.out_log
        ])
        
    def display(self):
        display(self.main_layout)
        
    def _run_merge(self, _):
        root_path = self.txt_input.value.strip()
        out_path = self.txt_output.value.strip()
        
        if not os.path.isdir(root_path):
            with self.out_log: print(f"❌ Input root not found: {root_path}")
            return
            
        if not out_path:
            out_path = os.path.join(root_path, "Aggregated")
            self.txt_output.value = out_path
            
        os.makedirs(out_path, exist_ok=True)
        
        # Gather Config
        config = {
            'cell': self.txt_cell.value,
            'cyto': self.txt_cyto.value,
            'nuc': self.txt_nuc.value,
            'img': self.txt_img.value
        }
        
        # Scan for well folders
        root = Path(root_path)
        well_folders = [f for f in root.iterdir() if f.is_dir() and f.name != os.path.basename(out_path)]
        
        if not well_folders:
            with self.out_log: print("❌ No subfolders found in input root.")
            return
            
        use_parallel = self.chk_parallel.value
        workers = self.int_workers.value if use_parallel else 1
        
        with self.out_log:
            print(f"🚀 Starting Data Merge ({len(well_folders)} wells)...")
            print(f"   Mode: {'Parallel (' + str(workers) + ')' if use_parallel else 'Sequential'}")
            
        try:
            sc_list = []
            img_list = []
            
            # Execution
            if use_parallel and workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(process_well, f, config): f.name for f in well_folders}
                    
                    for fut in tqdm(as_completed(futures), total=len(well_folders), desc="Merging Wells"):
                        res_sc, res_img = fut.result()
                        if res_sc is not None: sc_list.append(res_sc)
                        if res_img is not None: img_list.append(res_img)
            else:
                for f in tqdm(well_folders, desc="Merging Wells"):
                    res_sc, res_img = process_well(f, config)
                    if res_sc is not None: sc_list.append(res_sc)
                    if res_img is not None: img_list.append(res_img)
                    
            # Final Concatenation
            with self.out_log: print("💾 Saving Master Files...")
            
            if sc_list:
                master_sc = pd.concat(sc_list, ignore_index=True)
                sc_out = os.path.join(out_path, "master_single_cell.csv")
                master_sc.to_csv(sc_out, index=False)
                print(f"   ✅ Saved Single Cell Data ({len(master_sc)} rows) -> {sc_out}")
            else:
                print("   ⚠️ No Single Cell data merged.")

            if img_list:
                master_img = pd.concat(img_list, ignore_index=True)
                img_out = os.path.join(out_path, "master_image_level.csv")
                master_img.to_csv(img_out, index=False)
                print(f"   ✅ Saved Image Level Data ({len(master_img)} rows) -> {img_out}")
            else:
                 print("   ⚠️ No Image Level data merged.")
                 
            print("🎉 Done.")
            
        except Exception as e:
            with self.out_log: print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

# --- Worker Function ---

MERGE_KEYS = ["ImageNumber", "ObjectNumber"]

def load_and_prefix(file_path, prefix, is_metadata_source=False):
    df = pd.read_csv(file_path, low_memory=False)
    
    # Identify Metadata cols
    metadata_cols = [c for c in df.columns if c.startswith("Metadata_") or c in MERGE_KEYS]
    
    # Drop redundant metadata from non-primary sources
    if not is_metadata_source:
        drop_meta = [c for c in metadata_cols if c not in MERGE_KEYS]
        df = df.drop(columns=drop_meta, errors="ignore")
        
    # Rename feature cols
    # Recalculate metadata_cols in case we dropped some or just to be safe about what remains
    remaining_cols = df.columns
    # The pure feature columns are those NOT in metadata_cols (Wait, if we dropped them, they are gone)
    # The logic in original script:
    # feature_cols = [c for c in df.columns if c not in metadata_cols] 
    # BUT we need to be careful. 'metadata_cols' variable holds names from BEFORE drop. 
    # So if we dropped them, they are not in df.columns. 
    # If we kept them (Merge Keys), they are in df.columns.
    
    # Let's stick to original script logic, but be precise.
    # 1. Identify what IS a metadata col currently in DF
    current_meta = [c for c in df.columns if c.startswith("Metadata_") or c in MERGE_KEYS]
    
    # 2. Rename everything else
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c not in current_meta}
    df = df.rename(columns=rename_map)
    
    return df

def process_well(folder_path, config):
    """
    Processes a single well folder.
    Returns (single_cell_df, image_df). Both can be None if files missing.
    """
    well_id = folder_path.name
    
    # 1. Image Data
    img_f = folder_path / config['img']
    img_df = None
    if img_f.exists():
        try:
            img_df = pd.read_csv(img_f)
            img_df["Metadata_WellID"] = well_id
        except Exception:
            pass # corrupted or empty?

    # 2. Single Cell Data
    c_f = folder_path / config['cell']
    cy_f = folder_path / config['cyto']
    n_f = folder_path / config['nuc']
    sc_df = None
    
    if c_f.exists() and cy_f.exists() and n_f.exists():
        try:
            # Load and Merge
            # Cell is source of truth for Metadata
            df_c = load_and_prefix(c_f, "Cell", True)
            df_cy = load_and_prefix(cy_f, "Cytoplasm", False)
            df_n = load_and_prefix(n_f, "Nucleus", False)
            
            m = df_c.merge(df_cy, on=MERGE_KEYS).merge(df_n, on=MERGE_KEYS)
            m["Metadata_WellID"] = well_id
            sc_df = m
        except Exception as e:
            # print(f"Error in {well_id}: {e}")
            pass
            
    return sc_df, img_df
