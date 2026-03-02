import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from skimage import io, filters, exposure, util

class FocusAnalyzer:
    """
    Analyzes image focus using Laplacian variance and generates Maximum Intensity Projections (MIPs).
    """
    
    @staticmethod
    def calculate_focus_score(image: np.ndarray) -> float:
        """Calculates the variance of the Laplacian of the image."""
        if image.ndim > 2:
            image = image[..., 0] # Use first channel if multi-channel
        return np.var(filters.laplace(image))

    @staticmethod
    def filter_and_project(image_paths: List[Union[str, Path]], 
                           threshold_percent: float = 75.0) -> Tuple[np.ndarray, List[str]]:
        """
        Loads images, calculates focus scores, filters based on threshold, 
        and returns the MIP and list of used planes.
        """
        results = []
        for p in image_paths:
            p = Path(p)
            img = io.imread(p)
            score = FocusAnalyzer.calculate_focus_score(img)
            results.append({'path': p, 'image': img, 'score': score})
        
        # Determine threshold
        scores = [r['score'] for r in results]
        if not scores:
            raise ValueError("No images provided for focus analysis.")
            
        max_score = np.max(scores)
        threshold = max_score * (threshold_percent / 100.0)
        
        valid_planes = [r for r in results if r['score'] >= threshold]
        
        if not valid_planes:
            # Fallback: use the single best plane if all fail (unlikely with % threshold)
            valid_planes = [max(results, key=lambda x: x['score'])]
            
        # Create MIP
        # Stack images: (N, H, W) -> Max along axis 0
        stack = np.stack([r['image'] for r in valid_planes])
        mip = np.max(stack, axis=0)
        
        used_paths = [str(r['path']) for r in valid_planes]
        return mip, used_paths

class PreviewGenerator:
    """
    Generates RGB composite previews from individual channel images.
    """
    
    @staticmethod
    def normalize_image(image: np.ndarray, 
                        low_p: float = 2.0, 
                        high_p: float = 99.8) -> np.ndarray:
        """Normalizes image intensity to 0-1 range based on percentiles."""
        v_min, v_max = np.percentile(image, (low_p, high_p))
        return exposure.rescale_intensity(
            image, in_range=(v_min, v_max), out_range=(0.0, 1.0)
        ).astype(np.float32)

    @staticmethod
    def create_composite(channel_images: Dict[str, np.ndarray], 
                         channel_map: Dict[str, str]) -> np.ndarray:
        """
        Creates an RGB composite.
        
        Args:
            channel_images: Dict mapping channel ID (e.g. 'C1') to numpy image array.
            channel_map: Dict mapping channel ID to Color Name (e.g. {'C1': 'Blue'}).
                         Supported Colors: Red, Green, Blue, Cyan, Magenta, Yellow, Gray.
        """
        if not channel_images:
            raise ValueError("No images provided for composite.")
            
        # Get shape from first image
        first_img = next(iter(channel_images.values()))
        h, w = first_img.shape[:2]
        composite = np.zeros((h, w, 3), dtype=np.float32)
        
        # Color Weights
        COLORS = {
            'RED': (1,0,0), 'GREEN': (0,1,0), 'BLUE': (0,0,1),
            'CYAN': (0,1,1), 'MAGENTA': (1,0,1), 'YELLOW': (1,1,0),
            'GRAY': (1,1,1), 'GREY': (1,1,1)
        }
        
        for ch_id, img in channel_images.items():
            if ch_id in channel_map:
                color_name = channel_map[ch_id]
                # Handle tuple config if passed (Color, Intensity) - simplistic handling here
                if isinstance(color_name, (tuple, list)):
                    color_name = color_name[0]
                    
                weights = COLORS.get(color_name.upper(), (1,1,1))
                
                # Normalize
                norm_img = PreviewGenerator.normalize_image(img)
                
                # Add to composite
                for i in range(3):
                    if weights[i] > 0:
                        composite[:,:,i] += (norm_img * weights[i])
                        
        # Clip and convert to uint8
        final_rgb = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
        return final_rgb

