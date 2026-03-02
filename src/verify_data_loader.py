import sys
import os
import sqlite3
import pandas as pd
import shutil
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_loader_ui import DataLoaderUI

def create_dummy_db(path, table_name, data):
    with sqlite3.connect(path) as conn:
        pd.DataFrame(data).to_sql(table_name, conn, index=False)

def test_data_loader():
    print("Testing DataLoaderUI Logic...")
    
    base_dir = Path("test_loader_data")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()
    
    # 1. Setup Data
    cell_data = {"BoxID": [1, 2, 3], "Val": [0.1, 0.2, 0.3]}
    img_data = {"ImgID": [101, 102], "Meta": ["A", "B"]}
    
    # SQLite
    db_c = base_dir / "cells.db"
    db_i = base_dir / "images.db"
    create_dummy_db(db_c, "MyCells", cell_data)
    create_dummy_db(db_i, "MyImages", img_data)
    
    # CSV
    csv_c = base_dir / "cells.csv"
    csv_i = base_dir / "images.csv"
    pd.DataFrame(cell_data).to_csv(csv_c, index=False)
    pd.DataFrame(img_data).to_csv(csv_i, index=False)
    
    # 2. Test SQLite Logic
    print("  Testing SQLite Loading...")
    ui = DataLoaderUI()
    
    # Mocking UI interactions
    ui.txt_db_cell.value = str(db_c)
    ui._scan_tables(ui.txt_db_cell, ui.dd_table_cell)
    ui.dd_table_cell.value = "MyCells"
    
    ui.txt_db_img.value = str(db_i)
    ui._scan_tables(ui.txt_db_img, ui.dd_table_img)
    ui.dd_table_img.value = "MyImages"
    
    ui._load_sqlite(None)
    
    assert ui.df_cell is not None
    assert len(ui.df_cell) == 3
    assert ui.df_image is not None
    assert len(ui.df_image) == 2
    print("  ✅ SQLite Success")
    
    # 3. Test CSV Logic
    print("  Testing CSV Loading...")
    ui_csv = DataLoaderUI()
    ui_csv.txt_csv_cell.value = str(csv_c)
    ui_csv.txt_csv_img.value = str(csv_i)
    
    ui_csv._load_csv(None)
    
    assert ui_csv.df_cell is not None
    assert len(ui_csv.df_cell) == 3
    assert ui_csv.df_image is not None
    assert len(ui_csv.df_image) == 2
    print("  ✅ CSV Success")
    
    # Cleanup
    # shutil.rmtree(base_dir)

if __name__ == "__main__":
    test_data_loader()
