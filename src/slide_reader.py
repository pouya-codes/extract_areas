"""
Slide Reader Utility Module

This module provides utilities for reading regions from Whole Slide Images (WSI)
directly from file paths. This enables the path-based API where the frontend
only sends file paths and coordinates instead of transferring large image data.

Optimized for on-premise deployment where all services have access to the 
same filesystem (or shared storage like NFS/EFS).
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Union
from PIL import Image, ImageDraw
import numpy as np

# Try to import OpenSlide, fall back to a stub if not available
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("⚠ OpenSlide not installed. Install with: pip install openslide-python")


class SlideReader:
    """
    Utility class for reading regions from Whole Slide Images.
    
    Supports:
    - OpenSlide-compatible formats (.svs, .tiff, .ndpi, .vms, .mrxs, etc.)
    - Regular image files (.png, .jpg, .tif)
    
    Usage:
        reader = SlideReader(base_path="/path/to/slides")
        region = reader.read_region("subfolder/slide.svs", x=1000, y=2000, w=512, h=512)
    """
    
    def __init__(self, base_path: str = None, allowed_extensions: List[str] = None):
        """
        Initialize the SlideReader.
        
        Args:
            base_path: Base directory for slides. All slide paths will be resolved
                      relative to this. If None, absolute paths are expected.
            allowed_extensions: List of allowed file extensions. If None, defaults to
                               common WSI formats.
        """
        self.base_path = Path(base_path).expanduser() if base_path else None
        self.allowed_extensions = allowed_extensions or [
            '.svs', '.tiff', '.tif', '.ndpi', '.vms', '.mrxs', '.scn',
            '.bif', '.svslide', '.png', '.jpg', '.jpeg'
        ]
        self._cache = {}  # Cache for open slide handles
        self._cache_max_size = 10  # Maximum number of cached slides
    
    def resolve_path(self, slide_path: str) -> Path:
        """
        Resolve a slide path to an absolute path.
        
        Args:
            slide_path: Relative or absolute path to the slide.
            
        Returns:
            Absolute Path object.
            
        Raises:
            FileNotFoundError: If the slide doesn't exist.
            ValueError: If the path is outside the allowed base path (security).
        """
        path = Path(slide_path).expanduser()
        
        if self.base_path and not path.is_absolute():
            # Resolve relative to base path
            full_path = (self.base_path / path).resolve()
        else:
            full_path = path.resolve()
        
        # Security check: ensure path is within base_path if set
        if self.base_path:
            try:
                full_path.relative_to(self.base_path.resolve())
            except ValueError:
                raise ValueError(
                    f"Security violation: path '{slide_path}' is outside "
                    f"allowed base path '{self.base_path}'"
                )
        
        if not full_path.exists():
            raise FileNotFoundError(f"Slide not found: {full_path}")
        
        # Check extension
        if full_path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported file format: {full_path.suffix}. "
                f"Allowed: {self.allowed_extensions}"
            )
        
        return full_path
    
    def _get_slide(self, slide_path: Path):
        """Get a slide handle, using cache if available."""
        path_str = str(slide_path)
        
        if path_str in self._cache:
            return self._cache[path_str]
        
        # Evict oldest if cache is full
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            old_slide = self._cache.pop(oldest_key)
            old_slide.close()
        
        # Open new slide
        if OPENSLIDE_AVAILABLE and slide_path.suffix.lower() in ['.svs', '.tiff', '.tif', '.ndpi', '.vms', '.mrxs', '.scn', '.bif', '.svslide']:
            try:
                slide = openslide.OpenSlide(str(slide_path))
                self._cache[path_str] = slide
                return slide
            except openslide.OpenSlideError:
                # Fall back to PIL for non-WSI TIFF files
                pass
        
        # Use PIL for regular images
        return None  # Signal to use PIL
    
    def read_region(
        self,
        slide_path: str,
        x: int,
        y: int,
        width: int,
        height: int,
        level: int = 0,
        downsample: float = 1.0
    ) -> Image.Image:
        """
        Read a region from a slide.
        
        Args:
            slide_path: Path to the slide (relative to base_path or absolute).
            x: X coordinate of the top-left corner (at level 0).
            y: Y coordinate of the top-left corner (at level 0).
            width: Width of the region to read (at level 0).
            height: Height of the region to read (at level 0).
            level: Pyramid level to read from (0 = highest resolution).
            downsample: Additional downsampling factor to apply.
            
        Returns:
            PIL Image in RGB format.
        """
        full_path = self.resolve_path(slide_path)
        slide = self._get_slide(full_path)
        
        if slide is not None:
            # OpenSlide mode
            # Calculate output size based on level and downsample
            level_downsample = slide.level_downsamples[level] if level < len(slide.level_downsamples) else 1.0
            total_downsample = level_downsample * downsample
            
            out_width = int(width / total_downsample)
            out_height = int(height / total_downsample)
            
            # Read region at the specified level
            # Note: OpenSlide expects (x, y) at level 0, but (width, height) at target level
            level_width = int(width / level_downsample)
            level_height = int(height / level_downsample)
            
            region = slide.read_region((x, y), level, (level_width, level_height))
            region = region.convert("RGB")
            
            # Apply additional downsampling if needed
            if downsample > 1.0:
                region = region.resize((out_width, out_height), Image.Resampling.LANCZOS)
            
            return region
        else:
            # PIL mode for regular images
            with Image.open(full_path) as img:
                # Crop the region
                region = img.crop((x, y, x + width, y + height))
                
                # Apply downsampling if needed
                if downsample > 1.0:
                    out_width = int(width / downsample)
                    out_height = int(height / downsample)
                    region = region.resize((out_width, out_height), Image.Resampling.LANCZOS)
                
                return region.convert("RGB")
    
    def get_slide_dimensions(self, slide_path: str) -> Tuple[int, int]:
        """
        Get the dimensions of a slide at level 0.
        
        Args:
            slide_path: Path to the slide.
            
        Returns:
            Tuple of (width, height).
        """
        full_path = self.resolve_path(slide_path)
        slide = self._get_slide(full_path)
        
        if slide is not None:
            return slide.dimensions
        else:
            with Image.open(full_path) as img:
                return img.size
    
    def get_slide_info(self, slide_path: str) -> dict:
        """
        Get metadata about a slide.
        
        Args:
            slide_path: Path to the slide.
            
        Returns:
            Dictionary with slide information.
        """
        full_path = self.resolve_path(slide_path)
        slide = self._get_slide(full_path)
        
        if slide is not None:
            return {
                "path": str(full_path),
                "dimensions": slide.dimensions,
                "level_count": slide.level_count,
                "level_dimensions": list(slide.level_dimensions),
                "level_downsamples": list(slide.level_downsamples),
                "properties": dict(slide.properties),
                "format": "openslide"
            }
        else:
            with Image.open(full_path) as img:
                return {
                    "path": str(full_path),
                    "dimensions": img.size,
                    "level_count": 1,
                    "level_dimensions": [img.size],
                    "level_downsamples": [1.0],
                    "properties": {},
                    "format": "pil"
                }
    
    def close(self):
        """Close all cached slide handles."""
        for slide in self._cache.values():
            try:
                slide.close()
            except:
                pass
        self._cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def create_mask_from_points(
    points: List[Tuple[float, float]],
    size: Tuple[int, int]
) -> Image.Image:
    """
    Create a binary mask from polygon points.
    
    Args:
        points: List of (x, y) tuples defining the polygon.
        size: (width, height) of the output mask.
        
    Returns:
        PIL Image in "L" mode (grayscale), with 255 inside polygon, 0 outside.
    """
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Clamp points to image bounds
    clamped_points = [
        (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
        for x, y in points
    ]
    
    if len(clamped_points) >= 3:
        draw.polygon(clamped_points, outline=255, fill=255)
    
    return mask


# Global slide reader instance (initialized on first use)
_slide_reader: Optional[SlideReader] = None


def get_slide_reader(base_path: str = None) -> SlideReader:
    """
    Get or create the global slide reader instance.
    
    Args:
        base_path: Base path for slides. Only used when creating the instance.
        
    Returns:
        SlideReader instance.
    """
    global _slide_reader
    if _slide_reader is None:
        _slide_reader = SlideReader(base_path=base_path)
    return _slide_reader


def init_slide_reader(base_path: str) -> SlideReader:
    """
    Initialize (or reinitialize) the global slide reader with a specific base path.
    
    Args:
        base_path: Base directory containing slides.
        
    Returns:
        SlideReader instance.
    """
    global _slide_reader
    if _slide_reader is not None:
        _slide_reader.close()
    _slide_reader = SlideReader(base_path=base_path)
    return _slide_reader
