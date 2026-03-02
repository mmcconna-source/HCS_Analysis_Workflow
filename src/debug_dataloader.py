import sys
import os
import sqlite3
import pandas as pd
import shutil
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_loader_ui import DataLoaderUI

def create_tricky_db(path):
    data = {"Col A": [1, 2], "Col B": [3, 4]}
    with sqlite3.connect(path) as conn:
        # Table with space
        pd.DataFrame(data).to_sql("My Table", conn, index=False)

def test_debug():
    print("Debugging DataLoaderUI...")
    
    base_dir = Path("debug_loader_data")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()
    
    db_path = base_dir / "tricky.db"
    create_tricky_db(db_path)
    
    ui = DataLoaderUI()
    ui.txt_db_cell.value = str(db_path)
    ui.txt_db_img.value = str(db_path)
    
    # 1. Scan
    print("  Scanning...")
    ui._scan_tables(ui.txt_db_cell, ui.dd_table_cell)
    ui._scan_tables(ui.txt_db_img, ui.dd_table_img)
    
    # 2. Select "My Table"
    ui.dd_table_cell.value = "My Table"
    ui.dd_table_img.value = "My Table"
    
    # 3. Load
    print(f"  Loading from table 'My Table'...")
    ui._load_sqlite(None)
    
    if ui.df_cell is not None:
        print(f"  ✅ Load Success! Rows: {len(ui.df_cell)}")
    else:
        print("  ❌ Load Failed (df_cell is None)")
        
    # Check log output (we have to inspect the widget output implementation or capture stdout if it prints)
    # The UI prints to self.out_log widgets. 
    # We can inspect internal state if needed but the logic is main thing.

if __name__ == "__main__":
    test_debug()
