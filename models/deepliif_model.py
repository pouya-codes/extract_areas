"""
DeepLIIF Model Implementation

Wrapper for the DeepLIIF model that implements the BaseAIModel interface.
"""

from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import sys
from pathlib import Path

# Add DeepLIIF to path
sys.path.append("module/DeepLiff")

from models.base_model import BaseAIModel
from src.process_file import ImageProcessor


class DeepLIIFModel(BaseAIModel):
    """
    DeepLIIF model for IHC image analysis.
    
    This model processes IHC (Immunohistochemistry) images and provides
    cell segmentation and classification results.
    """
    
    def __init__(self):
        super().__init__(
            model_name="deepliif",
            model_version="1.0.0"
        )
        self.description = (
            "DeepLIIF: Deep-Learning Inferred Multiplex "
            "ImmunoFluorescence for IHC Image Analysis"
        )
        self.requires_mask = True
        self.processor = None
        # Tells the API layer to pass the region's microns-per-pixel and the
        # downsample it was read at, which together decide the tile size and
        # the cell size thresholds.
        self.accepts_mpp = True
        self.hyperparameters = {
            'eager_mode': False,
            'color_dapi': False,
            'color_marker': False,
            'tile_size': 0,
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DeepLIIF model.
        
        Args:
            config: Dict with keys:
                - model_dir: Path to model weights directory
                - tile_size: Size of tiles for processing (default: 256)
                - post_processing: Enable post-processing (default: True)
                - gpu_ids: List of GPU IDs to use (default: [])
        """
        model_dir = config.get('model_dir')
        if not model_dir:
            raise ValueError("model_dir is required in config")
        
        model_dir = str(Path(model_dir).absolute())
        
        tile_size = config.get('tile_size', 256)
        post_processing = config.get('post_processing', True)
        gpu_ids = config.get('gpu_ids', [])
        
        self.processor = ImageProcessor(
            model_dir,
            tile_size,
            post_processing,
            gpu_ids
        )
        
        self._is_initialized = True
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an IHC image with DeepLIIF.
        
        Args:
            image: Input IHC image (RGB)
            mask: Binary mask defining region of interest
            annotation_points: Not used by DeepLIIF
            hyperparameters: Optional overrides:
                - eager_mode: Use eager mode (single GPU)
                - color_dapi: Apply DAPI coloring
                - color_marker: Apply marker coloring
        
        Returns:
            Dict with:
                - processed_image: Segmentation overlay
                - scores: Cell counts and metrics
                - success: True if successful
                - error: Error message if failed
        """
        try:
            # Validate inputs
            is_valid, error = self.validate_input(image, mask)
            if not is_valid:
                return {
                    'success': False,
                    'error': error,
                    'str_result': f'Error: {error}'
                }
            
            # Merge hyperparameters
            params = {**self.hyperparameters}
            if hyperparameters:
                params.update(hyperparameters)
            
            # Preprocess: apply mask
            if mask:
                masked_image = self.preprocess_image(image, mask)
            else:
                masked_image = image
            
            # A tile covers a fixed amount of tissue at full resolution, so its
            # pixel size depends on the slide's own resolution; a downsampled
            # region keeps the pixel size and covers proportionally more
            # tissue. An explicit hyperparameter wins if one was given.
            mpp = params.get('mpp')
            downsample = params.get('downsample') or 1.0
            tile_size = int(params.get('tile_size') or 0)
            if tile_size <= 0:
                tile_size = self.processor.tile_size_for_mpp(mpp, downsample)

            # Run model inference
            processed_images, scores = self.processor.test_img(
                masked_image,
                eager_mode=params.get('eager_mode', False),
                color_dapi=params.get('color_dapi', False),
                color_marker=params.get('color_marker', False),
                tissue_mask=mask,
                tile_size=tile_size,
                mpp=mpp,
            )
            
            # Get segmentation overlay
            overlay_image = processed_images.get('SegRefined')
            if overlay_image is None:
                return {
                    'success': False,
                    'error': 'Model did not produce SegRefined output',
                    'str_result': 'Error: No segmentation output'
                }
            
            # Postprocess: white out masked regions
            if mask:
                result_image = self.postprocess_output(
                    overlay_image,
                    mask
                )
            else:
                result_image = overlay_image
            
            # Clean up scores (remove non-serializable data)
            if 'cell_coords' in scores:
                del scores['cell_coords']
            
            # This string is drawn on the slide next to the region, so keep it
            # to the numbers a pathologist actually reads off an IHC region.
            str_result_parts = []
            if 'num_total' in scores:
                str_result_parts.append(f"{scores['num_total']:,} cells")
            if 'num_pos' in scores and 'num_neg' in scores:
                str_result_parts.append(
                    f"{scores['num_pos']:,} pos / {scores['num_neg']:,} neg")
            if 'percent_pos' in scores:
                str_result_parts.append(f"{scores['percent_pos']}% positive")

            if not str_result_parts:
                str_result = "DeepLIIF segmentation complete"
            else:
                str_result = " · ".join(str_result_parts)

            return {
                'processed_image': result_image,
                'scores': scores,
                'str_result': str_result,
                'success': True,
                'metadata': {
                    'model': self.model_name,
                    'version': self.model_version,
                    'tile_size': tile_size,
                    'mpp': mpp,
                    'downsample': downsample,
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}",
                'str_result': f'Processing Error: {str(e)}'
            }
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """
        Get schema for DeepLIIF hyperparameters.
        """
        return {
            'eager_mode': {
                'type': 'bool',
                'default': False,
                'description': (
                    'Use eager mode for single GPU processing. '
                    'Set to True for single GPU, False for multi-GPU'
                )
            },
            'color_dapi': {
                'type': 'bool',
                'default': False,
                'description': 'Apply DAPI channel coloring to output'
            },
            'color_marker': {
                'type': 'bool',
                'default': False,
                'description': 'Apply marker channel coloring to output'
            },
            'tile_size': {
                'type': 'int',
                'default': 0,
                'min': 0,
                'max': 1024,
                'description': (
                    'Tile size in pixels. Leave at 0 to derive it from the '
                    'slide resolution, so that at full resolution each tile '
                    'covers ~128um of tissue, which is what the network was '
                    'trained on. Setting it by hand changes the speed and how '
                    'large the network expects cells to be.'
                )
            }
        }
    
    def cleanup(self) -> None:
        """Clean up DeepLIIF model resources."""
        if self.processor:
            # Add any GPU memory cleanup if needed
            del self.processor
            self.processor = None
