"""
Cell Image Tile Extraction Module

Core functions for extracting and processing single-cell image tiles from
CellProfiler output data. Designed for reuse in notebooks and web applications.

Usage:
    from tile_extraction import (
        NotebookConfig, ChannelMapping,
        extract_tile, extract_multichannel_tile,
        create_rgb_composite, export_tiles
    )
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTION CLASSES
# =============================================================================

class TileExtractionError(Exception):
    """Raised when tile cannot be extracted (e.g., coordinates out of bounds)."""
    pass

class ImageNotFoundError(Exception):
    """Raised when image file cannot be located."""
    pass

class InvalidChannelMappingError(Exception):
    """Raised when channel mapping configuration is invalid."""
    pass

class ConfigurationError(Exception):
    """Raised when notebook configuration is invalid or incomplete."""
    pass


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class NotebookConfig:
    """Configuration for the cell tile extraction pipeline.

    Attributes:
        csv_path: Path to cell-level CSV with measurements and UMAP coordinates
        image_base_path: Root directory containing the TIFF images, OR a dictionary mapping PlateIDs to directories.
        output_dir: Directory for saving extracted tiles
        tile_size: Size of square tile in pixels (tile_size x tile_size)
        x_column: Column name for cell X coordinate
        y_column: Column name for cell Y coordinate
        umap_x_column: Column name for UMAP X coordinate
        umap_y_column: Column name for UMAP Y coordinate
        well_column: Column name for well identifier
        field_column: Column name for field/site number
        plate_column: Column name for plate identifier (default: Metadata_PlateID)
        channel_names: List of channel names in order (C01, C02, ...)
        filename_pattern: Format string for image filenames. Keys: {well}, {field}, {channel} (1-based), {channel_name}

    Example:
        config = NotebookConfig(
            csv_path='./data/cells.csv',
            image_base_path='./images/', # or {'Plate1': './images/p1', 'Plate2': './images/p2'}
            channel_names=['DNA', 'KRT8', 'CMO', 'TP63', 'Phalloidin'],
            filename_pattern="{well}_F{field:04d}_T0001_Z0001_C{channel:02d}.tif"
        )
    """
    csv_path: str
    image_base_path: Any # str or Dict[str, str]
    output_dir: str = './output_tiles/'
    tile_size: int = 100

    # Column mappings
    x_column: str = 'Nucleus_AreaShape_Center_X'
    y_column: str = 'Nucleus_AreaShape_Center_Y'
    umap_x_column: str = 'UMAP1'
    umap_y_column: str = 'UMAP2'
    well_column: str = 'Metadata_WellID'
    field_column: str = 'Metadata_Field'
    plate_column: str = 'Metadata_PlateID'

    # Channel configuration: C01=DNA, C02=KRT8, C03=CMO, C04=TP63, C05=Phalloidin
    channel_names: List[str] = field(default_factory=lambda:
        ['DNA', 'KRT8', 'CMO', 'TP63', 'Phalloidin'])

    # Filename pattern for image resolution
    # Available keys: well, field, channel (1-based index)
    filename_pattern: str = "{well}_F{field:04d}_T0001_Z0001_C{channel:02d}.tif"
    
    # Bounding Box Columns (Optional, overrides x/y_column if present)
    bbox_min_x_column: Optional[str] = None
    bbox_min_y_column: Optional[str] = None
    bbox_max_x_column: Optional[str] = None
    bbox_max_y_column: Optional[str] = None

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not Path(self.csv_path).exists():
            raise ConfigurationError(f"CSV file not found: {self.csv_path}")
            
        # Validate image base path (str or dict)
        if isinstance(self.image_base_path, dict):
            for plate, path in self.image_base_path.items():
                if not Path(path).exists():
                     # Only warn if not exists? Or strict fail? 
                     # Strict fail is safer for now.
                     raise ConfigurationError(f"Image directory for plate '{plate}' not found: {path}")
        else:
            if not Path(self.image_base_path).exists():
                raise ConfigurationError(f"Image directory not found: {self.image_base_path}")
                
        if self.tile_size <= 0:
            raise ConfigurationError(f"tile_size must be positive, got {self.tile_size}")
        if len(self.channel_names) == 0:
            raise ConfigurationError("channel_names cannot be empty")


@dataclass
class ChannelMapping:
    """Configuration for mapping a single channel to RGB output.

    Attributes:
        channel_index: Index in the multichannel array (0-based)
        channel_name: Human-readable name (e.g., 'DNA')
        target_color: 'R', 'G', 'B', or None (exclude from composite)
        intensity_min: Minimum intensity (percentile or absolute)
        intensity_max: Maximum intensity (percentile or absolute)
        use_percentile: If True, min/max are percentiles; if False, absolute values
        gamma: Gamma correction (1.0 = no correction)

    Example:
        mapping = ChannelMapping(
            channel_index=0,
            channel_name='DNA',
            target_color='B',
            intensity_min=1.0,
            intensity_max=99.0
        )
    """
    channel_index: int
    channel_name: str
    target_color: Optional[str] = None
    intensity_min: float = 1.0
    intensity_max: float = 99.0
    use_percentile: bool = True
    gamma: float = 1.0

    def validate(self) -> None:
        """Validate channel mapping parameters.

        Raises:
            InvalidChannelMappingError: If parameters are invalid
        """
        if self.target_color not in [None, 'R', 'G', 'B', 'C', 'M', 'Y']:
            raise InvalidChannelMappingError(
                f"target_color must be 'R', 'G', 'B', 'C', 'M', 'Y', or None, got {self.target_color}")
        if self.use_percentile:
            if not (0 <= self.intensity_min <= 100 and 0 <= self.intensity_max <= 100):
                raise InvalidChannelMappingError(
                    "Percentile values must be between 0 and 100")
        if self.intensity_min >= self.intensity_max:
            raise InvalidChannelMappingError(
                f"intensity_min ({self.intensity_min}) must be less than intensity_max ({self.intensity_max})")
        if self.gamma <= 0:
            raise InvalidChannelMappingError(f"gamma must be positive, got {self.gamma}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_row_val_with_fallback(row: pd.Series, col_name: str) -> Any:
    """Retrieve value from row, checking col_name and Metadata_col_name."""
    if col_name in row:
        return row[col_name]
    elif f"Metadata_{col_name}" in row:
        return row[f"Metadata_{col_name}"]
    # Also check if col_name ALREADY has Metadata_ prefix and we should check without?
    # Usually the issue is config says "WellID" but DF has "Metadata_WellID", or vice versa.
    elif col_name.startswith("Metadata_") and col_name.replace("Metadata_", "") in row:
         return row[col_name.replace("Metadata_", "")]
         
    raise KeyError(f"Column '{col_name}' (or 'Metadata_' variant) not found in row.")


# =============================================================================
# IMAGE PATH RESOLUTION
# =============================================================================

def resolve_image_paths(row: pd.Series, config: NotebookConfig) -> Dict[str, Path]:
    """Map CSV row metadata to actual image file paths.

    Image naming pattern: {Well}_F{Field:04d}_T0001_Z0001_C{Channel:02d}.tif

    Args:
        row: Single row from cell CSV containing metadata
        config: Configuration object with channel names

    Returns:
        Dictionary mapping channel names to resolved file paths

    Raises:
        ImageNotFoundError: If resolved path does not exist
        ConfigurationError: If plate path not found in config mapping

    Example:
        paths = resolve_image_paths(df.iloc[0], config)
        # Returns: {'DNA': Path('B-02_F0001_T0001_Z0001_C01.tif'), ...}
    """
    well = get_row_val_with_fallback(row, config.well_column)
    field = int(get_row_val_with_fallback(row, config.field_column))

    # Resolve Base Path (Single str or Dict by Plate)
    if isinstance(config.image_base_path, dict):
        try:
            plate = get_row_val_with_fallback(row, config.plate_column)
        except KeyError:
             # Fallback: if dict has only one key, maybe use that? 
             # Or strict fail. Strict fail is better.
             raise ConfigurationError(f"Multi-plate config requires '{config.plate_column}' in data to resolve image path.")
             
        if plate in config.image_base_path:
            base_path = Path(config.image_base_path[plate])
        else:
            # Try matching str vs int just in case
             found = False
             for k, v in config.image_base_path.items():
                 if str(k) == str(plate):
                     base_path = Path(v)
                     found = True
                     break
             if not found:
                 raise ConfigurationError(f"Image path for Plate '{plate}' not defined in config.")
    else:
        base_path = Path(config.image_base_path)

    paths = {}

    for i, channel_name in enumerate(config.channel_names):
        channel_num = i + 1  # 1-based index
        
        try:
            filename = config.filename_pattern.format(
                well=well,
                field=field,
                channel=channel_num,
                channel_name=channel_name
            )
        except KeyError as e:
            raise ConfigurationError(f"Invalid placeholder in filename_pattern: {e}")
            
        filepath = base_path / filename

        if not filepath.exists():
            raise ImageNotFoundError(f"Image not found: {filepath}")

        paths[channel_name] = filepath

    return paths


# =============================================================================
# IMAGE CACHING
# =============================================================================

_image_cache: Dict[str, np.ndarray] = {}
_cache_max_size = 50


def load_image_cached(path: Path) -> np.ndarray:
    """Load image with caching to improve performance.

    Args:
        path: Path to TIFF image

    Returns:
        Image as numpy array
    """
    global _image_cache

    key = str(path)
    if key not in _image_cache:
        if len(_image_cache) >= _cache_max_size:
            oldest_key = next(iter(_image_cache))
            del _image_cache[oldest_key]
        _image_cache[key] = tifffile.imread(path)

    return _image_cache[key]


def clear_image_cache() -> None:
    """Clear the image cache to free memory."""
    global _image_cache
    _image_cache = {}


# =============================================================================
# TILE EXTRACTION
# =============================================================================

def extract_tile(image: np.ndarray, center_x: float, center_y: float,
                 tile_size: int, padding_mode: str = 'constant') -> np.ndarray:
    """Extract square tile centered on specified coordinates.

    Handles edge cases:
    - Cell near image boundary: pads with zeros (constant) or mirrors (reflect)
    - Non-integer coordinates: rounds to nearest pixel

    Args:
        image: 2D numpy array (single channel)
        center_x, center_y: Cell centroid coordinates in pixels
        tile_size: Side length of square tile in pixels
        padding_mode: 'constant' (zero-pad) or 'reflect' (mirror)

    Returns:
        2D numpy array of shape (tile_size, tile_size), dtype preserved

    Example:
        tile = extract_tile(image, 512.3, 256.7, tile_size=100)
    """
    # Round to nearest pixel
    cx = int(round(center_x))
    cy = int(round(center_y))

    half = tile_size // 2
    h, w = image.shape

    # Calculate extraction bounds
    y1 = cy - half
    y2 = cy + half + (tile_size % 2)
    x1 = cx - half
    x2 = cx + half + (tile_size % 2)

    # Calculate padding needed
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)

    # Clamp bounds to image
    y1_clamped = max(0, y1)
    y2_clamped = min(h, y2)
    x1_clamped = max(0, x1)
    x2_clamped = min(w, x2)

    # Extract region
    tile = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

    # Pad if necessary
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if padding_mode == 'constant':
            tile = np.pad(tile, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=0)
        elif padding_mode == 'reflect':
            tile = np.pad(tile, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='reflect')
        else:
            raise ValueError(f"Unknown padding_mode: {padding_mode}")

    assert tile.shape == (tile_size, tile_size), \
        f"Tile shape mismatch: expected ({tile_size}, {tile_size}), got {tile.shape}"

    return tile


def extract_multichannel_tile(row: pd.Series, config: NotebookConfig,
                               padding_mode: str = 'constant') -> np.ndarray:
    """Extract tiles from all channels for a single cell.

    Args:
        row: DataFrame row containing cell metadata and coordinates
        config: Notebook configuration
        padding_mode: 'constant' or 'reflect'

    Returns:
        3D numpy array of shape (n_channels, tile_size, tile_size)

    Raises:
        TileExtractionError: If extraction fails for any channel
    """
    
    if config.bbox_min_x_column and config.bbox_max_x_column:
         min_x = get_row_val_with_fallback(row, config.bbox_min_x_column)
         max_x = get_row_val_with_fallback(row, config.bbox_max_x_column)
         center_x = (min_x + max_x) / 2
    else:
         center_x = get_row_val_with_fallback(row, config.x_column)

    if config.bbox_min_y_column and config.bbox_max_y_column:
         min_y = get_row_val_with_fallback(row, config.bbox_min_y_column)
         max_y = get_row_val_with_fallback(row, config.bbox_max_y_column)
         center_y = (min_y + max_y) / 2
    else:
         center_y = get_row_val_with_fallback(row, config.y_column)

    try:
        paths = resolve_image_paths(row, config)
    except ImageNotFoundError as e:
        raise TileExtractionError(f"Could not resolve image paths: {e}")

    n_channels = len(config.channel_names)
    tiles = np.zeros((n_channels, config.tile_size, config.tile_size), dtype=np.uint16)

    for i, channel_name in enumerate(config.channel_names):
        try:
            image = load_image_cached(paths[channel_name])
            tiles[i] = extract_tile(image, center_x, center_y,
                                   config.tile_size, padding_mode)
        except Exception as e:
            raise TileExtractionError(
                f"Failed to extract tile for channel {channel_name}: {e}")

    return tiles


# =============================================================================
# RGB COMPOSITE GENERATION
# =============================================================================

def create_rgb_composite(multichannel_tile: np.ndarray,
                         mappings: List[ChannelMapping]) -> np.ndarray:
    """Create 8-bit RGB composite from multichannel 16-bit data.

    Processing steps:
    1. For each channel with a color assignment:
       a. Apply intensity scaling (percentile or absolute min/max)
       b. Apply gamma correction if specified
       c. Scale to 0-255 range
    2. Map channels to RGB based on target_color
    3. Handle channel blending (additive for same color target)
    4. Clip values to [0, 255] and convert to uint8

    Args:
        multichannel_tile: 3D array (n_channels, height, width), dtype=uint16
        mappings: List of ChannelMapping objects for each channel

    Returns:
        3D numpy array of shape (height, width, 3), dtype=uint8

    Example:
        mappings = [
            ChannelMapping(0, 'DNA', 'B'),
            ChannelMapping(1, 'KRT8', 'G'),
            ChannelMapping(4, 'Phalloidin', 'R')
        ]
        rgb = create_rgb_composite(tile, mappings)
    """
    n_channels, height, width = multichannel_tile.shape

    rgb = np.zeros((height, width, 3), dtype=np.float64)
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    
    # Map color codes to RGB channel indices
    # R=0, G=1, B=2
    # C=G+B, M=R+B, Y=R+G
    color_map = {
        'R': [0],
        'G': [1],
        'B': [2],
        'C': [1, 2],
        'M': [0, 2],
        'Y': [0, 1]
    }

    for mapping in mappings:
        if mapping.target_color is None:
            continue

        mapping.validate()

        channel_data = multichannel_tile[mapping.channel_index].astype(np.float64)

        if mapping.use_percentile:
            vmin = np.percentile(channel_data, mapping.intensity_min)
            vmax = np.percentile(channel_data, mapping.intensity_max)
        else:
            vmin = mapping.intensity_min
            vmax = mapping.intensity_max

        if vmax <= vmin:
            vmax = vmin + 1

        normalized = (channel_data - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)

        if mapping.gamma != 1.0:
            normalized = np.power(normalized, 1.0 / mapping.gamma)

        scaled = normalized * 255.0

        scaled = normalized * 255.0

        if mapping.target_color in color_map:
            for idx in color_map[mapping.target_color]:
                rgb[:, :, idx] += scaled

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


# =============================================================================
# BATCH EXPORT
# =============================================================================

def export_tiles(selected_df: pd.DataFrame, config: NotebookConfig,
                 channel_mappings: List[ChannelMapping],
                 output_format: str = 'png',
                 include_metadata: bool = True,
                 progress_callback=None) -> Dict[str, any]:
    """Batch export RGB tiles for all selected cells.

    File naming: {Metadata_WellID}_{Metadata_Field}_{ImageNumber}_{ObjectNumber}.{format}

    Args:
        selected_df: DataFrame subset of selected cells
        config: Notebook configuration
        channel_mappings: Channel-to-color assignments and intensity settings
        output_format: 'png' (lossless) or 'jpg' (smaller files)
        include_metadata: If True, save accompanying CSV with tile metadata
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        Dictionary with export statistics:
        - 'successful': Number of tiles exported
        - 'failed': Number of failed extractions
        - 'errors': List of error messages
        - 'output_path': Path to output directory
        - 'exported_files': List of exported filenames
    """
    from PIL import Image

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        'successful': 0,
        'failed': 0,
        'errors': [],
        'output_path': str(output_path),
        'exported_files': []
    }

    export_metadata = []
    total = len(selected_df)

    print(f"DEBUG: Starting export_tiles with {total} cells.")
    
    for i, (idx, row) in enumerate(selected_df.iterrows()):
        if i % 100 == 0:
            print(f"DEBUG: Processing tile {i+1}/{total}")
            
        if progress_callback:
            progress_callback(i, total)

        try:
            multichannel_tile = extract_multichannel_tile(row, config)
            rgb = create_rgb_composite(multichannel_tile, channel_mappings)

            well = get_row_val_with_fallback(row, config.well_column)
            field = int(get_row_val_with_fallback(row, config.field_column))
            
            # Handle Image/Object number with fallback
            try: img_num = int(get_row_val_with_fallback(row, 'ImageNumber'))
            except: img_num = 0
            
            try: obj_num = int(get_row_val_with_fallback(row, 'ObjectNumber'))
            except: obj_num = 0

            filename = f"{well}_{field}_{img_num}_{obj_num}_{i}.{output_format}"
            filepath = output_path / filename

            img = Image.fromarray(rgb)
            if output_format == 'jpg':
                img.save(filepath, quality=95)
            else:
                img.save(filepath)

            results['successful'] += 1
            results['exported_files'].append(filename)
            
            # Metadata logging
            meta_entry = {
                'filename': filename,
                'well': well,
                'field': field,
                'image_number': img_num,
                'object_number': obj_num,
                'umap_x': get_row_val_with_fallback(row, config.umap_x_column) if config.umap_x_column else 0,
                'umap_y': get_row_val_with_fallback(row, config.umap_y_column) if config.umap_y_column else 0
            }
            
            # Try to add center coords if available
            if config.x_column:
                try: meta_entry['center_x'] = get_row_val_with_fallback(row, config.x_column)
                except: pass
            if config.y_column:
                try: meta_entry['center_y'] = get_row_val_with_fallback(row, config.y_column)
                except: pass
                
            export_metadata.append(meta_entry)

        except Exception as e:
            results['failed'] += 1
            error_msg = f"Row {idx}: {str(e)}"
            results['errors'].append(error_msg)
            logger.warning(error_msg)

    if include_metadata and export_metadata:
        metadata_df = pd.DataFrame(export_metadata)
        metadata_path = output_path / 'tile_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")

    clear_image_cache()

    return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_validate_csv(config: NotebookConfig) -> pd.DataFrame:
    """Load cell-level CSV and validate required columns exist.

    Args:
        config: Notebook configuration object

    Returns:
        Validated pandas DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    logger.info(f"Loading CSV from {config.csv_path}...")

    df = pd.read_csv(config.csv_path)
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns):,} columns")

    required_columns = [
        'ImageNumber', 'ObjectNumber',
        config.well_column, config.field_column
    ]
    
    # Check for coordinates
    if config.bbox_min_x_column and config.bbox_max_x_column:
        required_columns.extend([config.bbox_min_x_column, config.bbox_max_x_column])
    else:
        required_columns.append(config.x_column)
        
    if config.bbox_min_y_column and config.bbox_max_y_column:
        required_columns.extend([config.bbox_min_y_column, config.bbox_max_y_column])
    else:
        required_columns.append(config.y_column)

    # UMAP columns are optional for initial load, but validated later if needed
    if config.umap_x_column in df.columns:
        required_columns.append(config.umap_x_column)
    if config.umap_y_column in df.columns:
        required_columns.append(config.umap_y_column)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    critical_cols = [config.well_column, config.field_column]
    # Add coord columns to critical list
    if config.bbox_min_x_column: critical_cols.append(config.bbox_min_x_column)
    else: critical_cols.append(config.x_column)
    
    if config.bbox_min_y_column: critical_cols.append(config.bbox_min_y_column)
    else: critical_cols.append(config.y_column)

    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count:,} NaN values")

    logger.info("CSV validation complete.")
    return df


def validate_image_paths(df: pd.DataFrame, config: NotebookConfig,
                         sample_size: int = 10) -> Dict[str, any]:
    """Validate image path resolution on a sample of rows.

    Args:
        df: DataFrame with cell data
        config: Notebook configuration
        sample_size: Number of rows to test

    Returns:
        Dictionary with validation results and any errors found
    """
    unique_images = df[[config.well_column, config.field_column]].drop_duplicates()
    n_sample = min(sample_size, len(unique_images))
    sample = unique_images.sample(n=n_sample, random_state=42)

    results = {
        'tested': n_sample,
        'successful': 0,
        'failed': 0,
        'errors': []
    }

    for _, row in sample.iterrows():
        try:
            paths = resolve_image_paths(row, config)
            results['successful'] += 1
        except ImageNotFoundError as e:
            results['failed'] += 1
            results['errors'].append(str(e))

    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_mappings(channel_names: List[str]) -> List[ChannelMapping]:
    """Create default channel mappings for common Cell Painting setup.

    Default: DNA=Blue, Phalloidin=Red, KRT8=Green

    Args:
        channel_names: List of channel names

    Returns:
        List of ChannelMapping objects
    """
    default_colors = {
        'DNA': 'B',
        'DAPI': 'B',
        'Hoechst': 'B',
        'KRT8': 'G',
        'GFP': 'G',
        'Phalloidin': 'R',
        'Actin': 'R',
    }

    mappings = []
    for i, name in enumerate(channel_names):
        mappings.append(ChannelMapping(
            channel_index=i,
            channel_name=name,
            target_color=default_colors.get(name, None),
            intensity_min=1.0,
            intensity_max=99.0,
            use_percentile=True,
            gamma=1.0
        ))

    return mappings


if __name__ == '__main__':
    # Example usage
    print("Cell Image Tile Extraction Module")
    print("=" * 40)
    print("Import this module to use the functions:")
    print("  from tile_extraction import NotebookConfig, extract_multichannel_tile, ...")
