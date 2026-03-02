import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple

# Try importing CellPose, handle if missing
try:
    from cellpose import models, io as cp_io
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: CellPose not installed. Segmentation features will be disabled.")

class PlateHeatmap:
    """
    Generates heatmaps for cell culture plates (96/384 format).
    """
    
    @staticmethod
    def plot_heatmap(df: pd.DataFrame, 
                     value_col: str, 
                     well_col: str = 'Metadata_WellID',
                     title: Optional[str] = None):
        """
        Plots a heatmap of the plate based on a specific column value.
        
        Args:
            df: DataFrame containing the data.
            value_col: Column to visualize (e.g. 'Count_Nuclei').
            well_col: Column containing Well IDs (e.g. 'A01').
        """
        if value_col not in df.columns:
            print(f"Error: Column {value_col} not found.")
            return

        # Extract Row/Col
        # Assumes WellID format "A01", "P24" etc.
        data = df.copy()
        try:
            data['Row'] = data[well_col].str[0]
            data['Col'] = data[well_col].str[1:].astype(int)
        except Exception as e:
            print(f"Error parsing Well IDs: {e}")
            return

        # Pivot
        heatmap_data = data.pivot(index='Row', columns='Col', values=value_col)
        
        # Sort index to ensure A, B, C order
        heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='viridis', linewidths=0.5, linecolor='gray', 
                    cbar_kws={'label': value_col})
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Plate Heatmap: {value_col}")
            
        plt.show()

class SegmentationGenerator:
    """
    Wrapper for CellPose segmentation.
    """
    
    def __init__(self, model_type: str = 'cyto', gpu: bool = False):
        if not CELLPOSE_AVAILABLE:
            raise ImportError("CellPose is not available in the environment.")
        
        self.model = models.Cellpose(gpu=gpu, model_type=model_type)
        
    def generate_masks(self, 
                       image_paths: List[Union[str, Path]], 
                       diameter: float = 30.0, 
                       channels: List[int] = [0, 0]) -> List[np.ndarray]:
        """
        Generates masks for a list of images.
        
        Args:
            image_paths: List of file paths.
            diameter: Approximate cell diameter (pixels).
            channels: [Cytoplasm, Nucleus] channels. 
                      [0,0] for grayscale, [2,3] for standardized channels.
        
        Returns:
            List of mask arrays (same shape as input images).
        """
        # Load images
        imgs = [cp_io.imread(str(p)) for p in image_paths]
        
        masks, flows, styles, diams = self.model.eval(imgs, diameter=diameter, channels=channels)
        
        return masks

    @staticmethod
    def save_masks(masks: List[np.ndarray], 
                   original_paths: List[Union[str, Path]], 
                   output_dir: Union[str, Path],
                   suffix: str = "_mask"):
        """Saves masks as TIFs or PNGs."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        for mask, p in zip(masks, original_paths):
            p = Path(p)
            save_name = out_path / f"{p.stem}{suffix}.png"
            # Save using matplotlib logic or skimage, here using creating a labeled image
            import skimage.io
            # Ensure mask is uint16 or suitable for saving
            skimage.io.imsave(save_name, mask.astype(np.uint16), check_contrast=False)
            
