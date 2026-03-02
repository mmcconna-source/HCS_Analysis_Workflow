import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Union

class PlateMapGenerator:
    """
    Tool for creating and exporting plate maps for high-content screening.
    Supports 96 and 384 well plates.
    """
    
    def __init__(self, plate_type: int = 96):
        if plate_type not in [96, 384]:
            raise ValueError("Only 96 or 384 well plates are supported.")
        
        self.plate_type = plate_type
        self.rows = 8 if plate_type == 96 else 16
        self.cols = 12 if plate_type == 96 else 24
        
        # Generate row labels (A, B, C...)
        self.row_labels = [string.ascii_uppercase[i] for i in range(self.rows)]
        # Generate col labels (1, 2, 3...)
        self.col_labels = [str(i+1) for i in range(self.cols)]
        
        # Initialize empty dataframe for the layout
        self.layout_df = self._initialize_dataframe()

    def _initialize_dataframe(self) -> pd.DataFrame:
        """Creates an empty DataFrame with all WellIDs."""
        wells = []
        for r in self.row_labels:
            for c in self.col_labels:
                # Format: A + 1 -> A01 (Standard convention)
                well_id = f"{r}{int(c):02d}"
                wells.append(well_id)
        
        df = pd.DataFrame({'WellID': wells})
        return df

    def assign_condition(self, 
                         column_name: str, 
                         value: Union[str, int, float], 
                         rows: Optional[List[str]] = None, 
                         cols: Optional[List[Union[str, int]]] = None,
                         wells: Optional[List[str]] = None):
        """
        Assigns a specific value to a condition column for a subset of wells.
        
        Args:
            column_name: Name of the metadata column (e.g., 'Drug', 'Concentration')
            value: The value to assign.
            rows: List of rows to apply to (e.g., ['A', 'B']).
            cols: List of columns to apply to (e.g., [1, 2, 3]).
            wells: Specific list of WellIDs (e.g., ['A01', 'B02']).
        """
        if column_name not in self.layout_df.columns:
            self.layout_df[column_name] = np.nan

        # Mask for selecting rows
        mask = pd.Series([False] * len(self.layout_df))

        if wells:
            mask = mask | self.layout_df['WellID'].isin(wells)
        
        if rows or cols:
            # Parse components of WellID for filtering
            current_rows = self.layout_df['WellID'].str[0]
            current_cols = self.layout_df['WellID'].str[1:].astype(int)
            
            row_mask = pd.Series([True] * len(self.layout_df))
            col_mask = pd.Series([True] * len(self.layout_df))
            
            if rows:
                row_mask = current_rows.isin(rows)
            if cols:
                # Ensure cols are integers for comparison
                cols_int = [int(c) for c in cols]
                col_mask = current_cols.isin(cols_int)
                
            mask = mask | (row_mask & col_mask)
            
        # Apply assignment
        self.layout_df.loc[mask, column_name] = value
        print(f"Assigned '{column_name}' = {value} to {mask.sum()} wells.")

    def visualize(self, column_name: str):
        """Visualizes the plate layout for a given column as a heatmap."""
        if column_name not in self.layout_df.columns:
            print(f"Column '{column_name}' not found.")
            return

        # Pivot for heatmap: Rows as index, Cols as columns
        pivot_data = self.layout_df.copy()
        pivot_data['Row'] = pivot_data['WellID'].str[0]
        pivot_data['Col'] = pivot_data['WellID'].str[1:].astype(int)
        
        heatmap_matrix = pivot_data.pivot(index='Row', columns='Col', values=column_name)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_matrix.isnull() == False, cmap="viridis", cbar=False, linewidths=1, linecolor='gray')
        
        # Overlay text
        for r_idx, r_label in enumerate(self.row_labels):
            for c_idx, c_label in enumerate(self.col_labels):
                val = heatmap_matrix.loc[r_label, int(c_label)]
                text = str(val) if pd.notnull(val) else ""
                plt.text(c_idx + 0.5, r_idx + 0.5, text, ha='center', va='center', fontsize=8, color='white')

        plt.title(f"Plate Map: {column_name}")
        plt.show()

    def export_csv(self, filepath: Union[str, Path]):
        """Exports the layout to a CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.layout_df.to_csv(path, index=False)
        print(f"Plate map exported to: {path}")

# Example Usage
if __name__ == "__main__":
    # Create a 96-well plate
    plate = PlateMapGenerator(plate_type=96)
    
    # Assign Controls
    plate.assign_condition('Type', 'Control', cols=[1, 12])
    
    # Assign Drug A
    plate.assign_condition('Type', 'Drug_A', cols=range(2, 7))
    plate.assign_condition('Concentration', 10, rows=['A', 'B'], cols=range(2, 7))
    
    # Export
    # plate.export_csv("example_plate_map.csv")
