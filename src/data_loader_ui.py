import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import sqlite3
import os
import glob
from pathlib import Path

class DataLoaderUI:
    """
    UI for loading Cell and Image level data from SQLite databases or CSV files.
    Supports single file or batch loading (folder + pattern).
    """
    
    def __init__(self):
        self.df_cell = None
        self.df_image = None
        self.chk_batch_mode = widgets.Checkbox(value=False, description="Batch Mode (Load Multiple Files)")
        self._create_ui()
        
    def _create_ui(self):
        # --- Tab 1: SQLite ---
        self.txt_db_cell = widgets.Text(placeholder="Path to Cell .sqlite (or Folder if Batch)", description="Cell Path:")
        self.txt_db_img = widgets.Text(placeholder="Path to Image .sqlite (or Folder if Batch)", description="Image Path:")
        
        self.txt_pattern_cell = widgets.Text(value="*.sqlite", description="Pattern:", layout=widgets.Layout(width='150px'), disabled=True)
        self.txt_pattern_img = widgets.Text(value="*.sqlite", description="Pattern:", layout=widgets.Layout(width='150px'), disabled=True)
        
        self.btn_scan_cell = widgets.Button(description="Scan Tables", button_style='info', layout=widgets.Layout(width='100px'))
        self.dd_table_cell = widgets.Dropdown(description="Table:", options=[], disabled=True)
        
        self.btn_scan_img = widgets.Button(description="Scan Tables", button_style='info', layout=widgets.Layout(width='100px'))
        self.dd_table_img = widgets.Dropdown(description="Table:", options=[], disabled=True)
        
        self.btn_load_sqlite = widgets.Button(description="Load SQLite Data", button_style='success', layout=widgets.Layout(width='100%'))
        
        # Events
        self.chk_batch_mode.observe(self._toggle_batch, names='value')
        self.btn_scan_cell.on_click(lambda b: self._scan_tables(self.txt_db_cell, self.dd_table_cell))
        self.btn_scan_img.on_click(lambda b: self._scan_tables(self.txt_db_img, self.dd_table_img))
        self.btn_load_sqlite.on_click(self._load_sqlite)
        
        tab_sqlite = widgets.VBox([
            self.chk_batch_mode,
            widgets.HTML("<b>Cell Level Data</b>"),
            widgets.HBox([self.txt_db_cell, self.txt_pattern_cell]),
            widgets.HBox([self.btn_scan_cell, self.dd_table_cell]),
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Image Level Data</b>"),
            widgets.HBox([self.txt_db_img, self.txt_pattern_img]),
            widgets.HBox([self.btn_scan_img, self.dd_table_img]),
            widgets.HTML("<br>"),
            self.btn_load_sqlite
        ], layout=widgets.Layout(padding='10px'))
        
        # --- Tab 2: CSV ---
        self.txt_csv_cell = widgets.Text(placeholder="Path to Cell.csv (or Folder)", description="Cell Path:")
        self.txt_csv_img = widgets.Text(placeholder="Path to Image.csv (or Folder)", description="Image Path:")
        
        # Re-use patterns for CSV tab but different instances
        self.txt_csv_pattern_cell = widgets.Text(value="*Cell*.csv", description="Pattern:", layout=widgets.Layout(width='150px'), disabled=True)
        self.txt_csv_pattern_img = widgets.Text(value="*Image*.csv", description="Pattern:", layout=widgets.Layout(width='150px'), disabled=True)

        self.btn_load_csv = widgets.Button(description="Load CSV Data", button_style='success', layout=widgets.Layout(width='100%'))
        self.btn_load_csv.on_click(self._load_csv)
        
        tab_csv = widgets.VBox([
            widgets.HTML("<b>Batch Mode is controlled by the checkbox in the SQLite tab (shared).</b>"),
            widgets.HTML("<b>Cell Level Data</b>"),
            widgets.HBox([self.txt_csv_cell, self.txt_csv_pattern_cell]),
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Image Level Data</b>"),
            widgets.HBox([self.txt_csv_img, self.txt_csv_pattern_img]),
            widgets.HTML("<br>"),
            self.btn_load_csv
        ], layout=widgets.Layout(padding='10px'))
        
        # --- Main Layout ---
        self.tabs = widgets.Tab(children=[tab_sqlite, tab_csv])
        self.tabs.set_title(0, "SQLite Loader")
        self.tabs.set_title(1, "CSV Loader")
        
        self.out_log = widgets.Output(layout={'height': '150px', 'overflow_y': 'scroll', 'border': '1px solid #ccc', 'margin': '10px 0'})
        
        self.main_layout = widgets.VBox([
            widgets.HTML("<h2>Data Loader</h2>"),
            self.tabs,
            widgets.HTML("<b>Log:</b>"),
            self.out_log
        ])
        
    def display(self):
        display(self.main_layout)
        
    def _toggle_batch(self, change):
        is_batch = change['new']
        # Enable patterns
        self.txt_pattern_cell.disabled = not is_batch
        self.txt_pattern_img.disabled = not is_batch
        self.txt_csv_pattern_cell.disabled = not is_batch
        self.txt_csv_pattern_img.disabled = not is_batch
        
        if is_batch:
            self.btn_scan_cell.description = "Scan 1st File"
            self.btn_scan_img.description = "Scan 1st File"
        else:
            self.btn_scan_cell.description = "Scan Tables"
            self.btn_scan_img.description = "Scan Tables"

    def _get_files(self, path, pattern):
        """Helper to get file list based on mode."""
        if self.chk_batch_mode.value:
            # path is a directory
            if not os.path.isdir(path):
                return []
            search_path = os.path.join(path, pattern)
            return sorted(glob.glob(search_path))
        else:
            # path is a file
            if os.path.isfile(path):
                return [path]
            return []

    def _scan_tables(self, txt_widget, dd_widget):
        path = txt_widget.value.strip()
        pattern = self.txt_pattern_cell.value if txt_widget == self.txt_db_cell else self.txt_pattern_img.value
        
        files = self._get_files(path, pattern)
        
        if not files:
            with self.out_log: print(f"❌ No files found in {path} with pattern {pattern}" if self.chk_batch_mode.value else f"❌ File not found: {path}")
            return
            
        target_file = files[0] # Scan the first file to get schema
        
        try:
            with sqlite3.connect(target_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [r[0] for r in cursor.fetchall()]
                
            if tables:
                dd_widget.options = tables
                dd_widget.disabled = False
                # Auto-select if obvious
                default = next((t for t in tables if 'cell' in t.lower() or 'object' in t.lower()), None)
                if default: dd_widget.value = default
                
                with self.out_log: print(f"✅ Found {len(tables)} tables in {os.path.basename(target_file)} (and {len(files)-1} other files)")
            else:
                with self.out_log: print(f"⚠️ No tables found in {os.path.basename(target_file)}")
                
        except Exception as e:
            with self.out_log: print(f"❌ Error scanning DB: {e}")

    def _load_sqlite(self, _):
        cell_path = self.txt_db_cell.value.strip()
        cell_tbl = self.dd_table_cell.value
        cell_pattern = self.txt_pattern_cell.value
        
        img_path = self.txt_db_img.value.strip()
        img_tbl = self.dd_table_img.value
        img_pattern = self.txt_pattern_img.value
        
        if not cell_path or not cell_tbl:
            with self.out_log: print("❌ Please select Cell Database/Path and Table.")
            return
            
        if not img_path or not img_tbl:
            with self.out_log: print("❌ Please select Image Database/Path and Table.")
            return
            
        with self.out_log: print("⏳ Loading SQLite Data...")
        
        try:
            # Load Cell
            cell_files = self._get_files(cell_path, cell_pattern)
            if not cell_files:
                with self.out_log: print(f"❌ No Cell files found.")
                return

            cell_frames = []
            for f in cell_files:
                with sqlite3.connect(f) as conn:
                    df = pd.read_sql_query(f'SELECT * FROM "{cell_tbl}"', conn)
                    # Optional: Add Source File Column
                    df['SourceFile'] = os.path.basename(f)
                    cell_frames.append(df)
            self.df_cell = pd.concat(cell_frames, ignore_index=True)
                
            # Load Image
            img_files = self._get_files(img_path, img_pattern)
            if not img_files:
                with self.out_log: print(f"❌ No Image files found.")
                return

            img_frames = []
            for f in img_files:
                with sqlite3.connect(f) as conn:
                    df = pd.read_sql_query(f'SELECT * FROM "{img_tbl}"', conn)
                    df['SourceFile'] = os.path.basename(f)
                    img_frames.append(df)
            self.df_image = pd.concat(img_frames, ignore_index=True)
                
            self._post_load_check()
            
        except Exception as e:
            with self.out_log: 
                print(f"❌ SQLite Load Error: {e}")
                import traceback
                traceback.print_exc()

    def _load_csv(self, _):
        cell_path = self.txt_csv_cell.value.strip()
        cell_pattern = self.txt_csv_pattern_cell.value
        
        img_path = self.txt_csv_img.value.strip()
        img_pattern = self.txt_csv_pattern_img.value
        
        with self.out_log: print("⏳ Loading CSV Data...")
        
        try:
            # Load Cell
            cell_files = self._get_files(cell_path, cell_pattern)
            if not cell_files:
                 with self.out_log: print(f"❌ No Cell files found.")
                 return
            
            cell_frames = []
            for f in cell_files:
                try:
                    df = pd.read_csv(f, low_memory=False)
                except pd.errors.ParserError:
                     df = pd.read_csv(f, low_memory=False, engine='python')
                
                df['SourceFile'] = os.path.basename(f)
                cell_frames.append(df)
            self.df_cell = pd.concat(cell_frames, ignore_index=True)

            # Load Image
            img_files = self._get_files(img_path, img_pattern)
            if not img_files:
                 with self.out_log: print(f"❌ No Image files found.")
                 return
            
            img_frames = []
            for f in img_files:
                try:
                    df = pd.read_csv(f, low_memory=False)
                except pd.errors.ParserError:
                     df = pd.read_csv(f, low_memory=False, engine='python')

                df['SourceFile'] = os.path.basename(f)
                img_frames.append(df)
            self.df_image = pd.concat(img_frames, ignore_index=True)
            
            self._post_load_check()

        except Exception as e:
            with self.out_log: 
                print(f"❌ CSV Load Error: {e}")
                import traceback
                traceback.print_exc()

    def _post_load_check(self):
        c_shape = self.df_cell.shape if self.df_cell is not None else "None"
        i_shape = self.df_image.shape if self.df_image is not None else "None"
        
        print(f"✅ Data Loaded Successfully.")
        print(f"   Cell Data: {c_shape}")
        if self.df_cell is not None:
             print(f"   Columns: {list(self.df_cell.columns)[:5]} ...")
        print(f"   Image Data: {i_shape}")
        
        print("\n👇 RUN THIS IN THE NEXT CELL TO ACCESS DATA 👇")
        print("df_cell, df_image = loader.get_data()")

        
    def get_data(self):
        """Returns tuple (df_cell, df_image)"""
        return self.df_cell, self.df_image
