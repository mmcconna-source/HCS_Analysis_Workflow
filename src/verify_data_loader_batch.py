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

def test_batch_loader():
    print("Testing DataLoaderUI Batch Logic...")
    
    base_dir = Path("test_batch_data")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()
    
    # 1. Setup Data - Batch 1
    cell_data_1 = {"BoxID": [1, 2], "Val": [0.1, 0.2]}
    img_data_1 = {"ImgID": [101], "Meta": ["A"]}
    
    # Batch 2
    cell_data_2 = {"BoxID": [3, 4], "Val": [0.3, 0.4]}
    img_data_2 = {"ImgID": [102], "Meta": ["B"]}
    
    # SQLite Setup
    db_c1 = base_dir / "cells_1.sqlite"
    db_c2 = base_dir / "cells_2.sqlite"
    create_dummy_db(db_c1, "MyCells", cell_data_1)
    create_dummy_db(db_c2, "MyCells", cell_data_2)
    
    db_i1 = base_dir / "images_1.sqlite"
    db_i2 = base_dir / "images_2.sqlite"
    create_dummy_db(db_i1, "MyImages", img_data_1)
    create_dummy_db(db_i2, "MyImages", img_data_2)
    
    # CSV Setup
    csv_c1 = base_dir / "cells_1.csv"
    csv_c2 = base_dir / "cells_2.csv"
    pd.DataFrame(cell_data_1).to_csv(csv_c1, index=False)
    pd.DataFrame(cell_data_2).to_csv(csv_c2, index=False)
    
    csv_i1 = base_dir / "images_1.csv"
    csv_i2 = base_dir / "images_2.csv"
    pd.DataFrame(img_data_1).to_csv(csv_i1, index=False)
    pd.DataFrame(img_data_2).to_csv(csv_i2, index=False)
    
    # 2. Test SQLite Batch Logic
    print("  Testing SQLite Batch Loading...")
    ui = DataLoaderUI()
    ui.chk_batch_mode.value = True
    
    # Mocking UI interactions
    ui.txt_db_cell.value = str(base_dir)
    ui.txt_pattern_cell.value = "cells_*.sqlite"
    ui._scan_tables(ui.txt_db_cell, ui.dd_table_cell)
    ui.dd_table_cell.value = "MyCells"
    
    ui.txt_db_img.value = str(base_dir)
    ui.txt_pattern_img.value = "images_*.sqlite"
    ui._scan_tables(ui.txt_db_img, ui.dd_table_img)
    ui.dd_table_img.value = "MyImages"
    
    ui._load_sqlite(None)
    
    assert ui.df_cell is not None
    assert len(ui.df_cell) == 4 # 2 from each file
    assert 'SourceFile' in ui.df_cell.columns
    
    assert ui.df_image is not None
    assert len(ui.df_image) == 2 # 1 from each file
    print("  ✅ SQLite Batch Success")
    
    # 3. Test CSV Batch Logic
    print("  Testing CSV Batch Loading...")
    ui_csv = DataLoaderUI()
    ui_csv.chk_batch_mode.value = True
    
    ui_csv.txt_csv_cell.value = str(base_dir)
    ui_csv.txt_csv_pattern_cell.value = "cells_*.csv"
    
    ui_csv.txt_csv_img.value = str(base_dir)
    ui_csv.txt_csv_pattern_img.value = "images_*.csv"
    
    ui_csv._load_csv(None)
    
    assert ui_csv.df_cell is not None
    assert len(ui_csv.df_cell) == 4
    assert 'SourceFile' in ui_csv.df_cell.columns
    
    assert ui_csv.df_image is not None
    assert len(ui_csv.df_image) == 2
    print("  ✅ CSV Batch Success")
    
    # Cleanup
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    test_batch_loader()
