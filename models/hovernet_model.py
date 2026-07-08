"""
HoVer-Net Model Implementation

Wrapper for the HoVer-Net model that implements the BaseAIModel interface.
HoVer-Net performs nucleus segmentation and classification from H&E-stained histology images.

Repository: https://github.com/vqdang/hover_net
Paper: "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei 
        in Multi-Tissue Histology Images"
"""

from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import OrderedDict
import sys
from pathlib import Path
import json

from models.base_model import BaseAIModel


def load_model_configs_from_file():
    """Load HoVer-Net model configurations from config.json file."""
    config_path = Path(__file__).parent.parent / 'config.json'
    
    if not config_path.exists():
        # Fallback to default configs if file doesn't exist
        return {
            'pannuke': {
                'checkpoint': 'models/hovernet_checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar',
                'type_info': 'models/hovernet_checkpoints/type_info_pannuke.json',
                'nr_types': 6,
                'mode': 'fast',
                'description': 'PanNuke - 6 nucleus types'
            }
        }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('hovernet_models', {})
    except Exception as e:
        print(f"Warning: Could not load hovernet_models from config.json: {e}")
        # Return default fallback
        return {
            'pannuke': {
                'checkpoint': 'models/hovernet_checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar',
                'type_info': 'models/hovernet_checkpoints/type_info_pannuke.json',
                'nr_types': 6,
                'mode': 'fast',
                'description': 'PanNuke - 6 nucleus types'
            }
        }


# Load model configurations from config.json at module import time
MODEL_CONFIGS = load_model_configs_from_file()


class HoVerNetModel(BaseAIModel):
    """
    HoVer-Net model for nucleus segmentation and classification.

    This model processes H&E-stained histology images and provides:
    - Nucleus instance segmentation
    - Nucleus type classification (if trained)
    - Nucleus centroids and boundaries
    """

    def __init__(self):
        super().__init__(
            model_name="hovernet",
            model_version="1.0.0"
        )
        self.description = (
            "HoVer-Net: Simultaneous Segmentation and Classification "
            "of Nuclei in Multi-Tissue Histology Images"
        )
        self.requires_mask = False  # Can work without mask
        self.model = None
        self.device = None
        self.nr_types = None
        self.type_info_dict = None
        self.patch_input_shape = None
        self.patch_output_shape = None
        self.current_variant = None  # Track currently loaded variant

        # Default hyperparameters
        self.hyperparameters = {
            'model_variant': 'pannuke',  # Default model variant
            'model_mode': 'fast',  # 'original' or 'fast'
            'batch_size': 32,
            'draw_centroids': True,
            'min_nucleus_size': 10
        }

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the HoVer-Net model.

        Args:
            config: Dict with keys:
                - model_variant: Model variant name 
                  ('pannuke', 'consep', 'monusac', 'kumar')
                  OR provide custom paths:
                - model_path: Path to checkpoint file (.tar)
                - model_mode: 'original' or 'fast' (270x270 or 256x256)
                - nr_types: Number of nucleus types (0 for segmentation)
                - type_info_path: Optional path to type info JSON file
                - device: 'cuda' or 'cpu'
                - gpu_ids: List of GPU IDs (default: [0])
                - batch_size: Batch size for inference (default: 8)
        """
        # Reload config from file to get fresh values
        current_model_configs = load_model_configs_from_file()
        
        # Check if using a predefined model variant
        model_variant = config.get('model_variant')
        if model_variant and model_variant in current_model_configs:
            # Use predefined configuration
            variant_config = current_model_configs[model_variant]
            model_path = variant_config['checkpoint']
            self.model_mode = variant_config['mode']
            self.nr_types = variant_config['nr_types']
            type_info_path = variant_config['type_info']
            
            # Update hyperparameters
            self.hyperparameters['model_variant'] = model_variant
            self.hyperparameters['model_mode'] = self.model_mode
            self.current_variant = model_variant  # Track loaded variant
        else:
            # Use custom configuration
            model_path = config.get('model_path')
            if not model_path:
                raise ValueError(
                    "Either 'model_variant' or 'model_path' "
                    "is required in config"
                )
            self.model_mode = config.get('model_mode', 'fast')
            self.nr_types = config.get('nr_types', 0)
            type_info_path = config.get('type_info_path')
            self.current_variant = None  # Custom model, not a variant

        model_path = str(Path(model_path).absolute())
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}")

        # Update hyperparameters from config
        if 'batch_size' in config:
            self.hyperparameters['batch_size'] = config['batch_size']
        
        if self.nr_types == 0:
            self.nr_types = None  # HoVer-Net uses None for segmentation

        # Set patch shapes based on model mode
        if self.model_mode == 'fast':
            self.patch_input_shape = 256
            self.patch_output_shape = 164
        else:  # original
            self.patch_input_shape = 270
            self.patch_output_shape = 80

        # Device configuration
        device_name = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_name)

        # Load type information if provided
        if type_info_path and self.nr_types is not None:
            type_info_path = str(Path(type_info_path).absolute())
            if Path(type_info_path).exists():
                with open(type_info_path, 'r') as f:
                    self.type_info_dict = json.load(f)
                    self.type_info_dict = {
                        int(k): (v[0], tuple(v[1]))
                        for k, v in self.type_info_dict.items()
                    }

        # Load model
        self._load_model(model_path, config.get('gpu_ids', [0]))

        self._is_initialized = True

    def _load_model(self, model_path: str, gpu_ids: List[int]) -> None:
        """Load HoVer-Net model from checkpoint."""
        # Import HoVer-Net model architecture
        # Note: You'll need to add HoVer-Net code to your project
        # For now, we'll use a simplified loading approach

        try:
            # Dynamically import HoVer-Net modules.
            # We vend the upstream HoVer-Net repo under:
            #   extract_areas/module/hovernet
            # (moved from extract_areas/models/hovernet)
            hovernet_repo = (
                Path(__file__).resolve().parent.parent / 'module' / 'hovernet'
            )
            if not hovernet_repo.exists():
                raise FileNotFoundError(
                    f"HoVer-Net repository not found at {hovernet_repo}. "
                    "Clone: git clone https://github.com/vqdang/hover_net.git"
                )
            
            # Add hovernet to path at the beginning
            hovernet_repo_str = str(hovernet_repo.absolute())
            if hovernet_repo_str not in sys.path:
                sys.path.insert(0, hovernet_repo_str)
            
            # Clear cached modules that might conflict
            modules_to_save = {}
            for mod_name in ['models', 'config']:
                if mod_name in sys.modules:
                    modules_to_save[mod_name] = sys.modules[mod_name]
                    del sys.modules[mod_name]
            
            try:
                # Import from hovernet
                from models.hovernet import net_desc
                create_model = net_desc.create_model
            finally:
                # Restore our modules
                for mod_name, mod_obj in modules_to_save.items():
                    sys.modules[mod_name] = mod_obj

            # Create model
            model_args = {
                'nr_types': self.nr_types,
                'mode': self.model_mode
            }
            net = create_model(**model_args)

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('desc', checkpoint)

            # Convert state dict if needed
            state_dict = self._convert_pytorch_checkpoint(state_dict)

            # Load weights
            net.load_state_dict(state_dict, strict=True)

            # Multi-GPU support
            if len(gpu_ids) > 1 and torch.cuda.is_available():
                net = torch.nn.DataParallel(net, device_ids=gpu_ids)

            net = net.to(self.device)
            net.eval()

            self.model = net

        except Exception as e:
            raise RuntimeError(f"Failed to load HoVer-Net model: {str(e)}")

    def _convert_pytorch_checkpoint(self, state_dict: Dict) -> Dict:
        """Convert checkpoint format if needed."""
        # Handle different checkpoint formats
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a histology image with HoVer-Net.

        Args:
            image: Input H&E image (RGB)
            mask: Optional binary mask defining region of interest
            annotation_points: Not used by HoVer-Net
            hyperparameters: Optional overrides:
                - model_mode: 'original' or 'fast'
                - batch_size: Batch size for inference
                - draw_centroids: Draw nucleus centroids
                - min_nucleus_size: Minimum nucleus area (pixels)

        Returns:
            Dict with:
                - processed_image: Segmentation overlay on original image
                - scores: Nucleus counts and type statistics
                - str_result: Summary string for display
                - success: True if successful
                - error: Error message if failed
                - metadata: Additional processing information
        """
        try:
            # Validate model is initialized
            if not self._is_initialized or self.model is None:
                return {
                    'success': False,
                    'error': 'Model not initialized',
                    'str_result': 'Error: Model not initialized'
                }

            # Merge hyperparameters
            params = {**self.hyperparameters}
            if hyperparameters:
                params.update(hyperparameters)
            
            # Check if model variant has changed - reinitialize if needed
            requested_variant = params.get('model_variant')
            
            # Reload config from file to get fresh values
            current_model_configs = load_model_configs_from_file()
            
            if (
                requested_variant
                and requested_variant != self.current_variant
                and requested_variant in current_model_configs
            ):

                print(
                    "Switching model variant from "
                    f"{self.current_variant} to {requested_variant}"
                )
                
                # Build new config - initialize will use fresh config
                reinit_config = {
                    'model_variant': requested_variant,
                    'device': str(self.device),
                    'gpu_ids': [0],  # Use same GPU setup
                    'batch_size': params.get('batch_size', 32)
                }
                
                # Cleanup old model
                self.cleanup()
                
                # Reinitialize with new variant
                self.initialize(reinit_config)
                
                print(f"✓ Switched to {requested_variant} variant")
            
            # Auto-adjust batch size for GPU memory
            # Reduce batch size for large images to prevent OOM
            img_pixels = image.width * image.height
            if img_pixels > 2000000:  # Images larger than ~1414x1414
                # Use smaller batch size for large images
                if 'batch_size' not in (hyperparameters or {}):
                    params['batch_size'] = min(params.get('batch_size', 32), 8)
                    print(f"Auto-adjusted batch_size to {params['batch_size']} for large image")

            # Convert PIL to numpy
            img_np = np.array(image)
            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

            # Apply mask if provided
            if mask:
                mask_np = np.array(mask)
                if mask_np.ndim == 3:
                    mask_np = mask_np[:, :, 0]
                # Apply mask
                img_np[mask_np == 0] = 255  # White out non-masked areas

            # Run inference
            # save image to disk for debugging
            cv2.imwrite("debug_input_image.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            # Try inference with automatic batch size reduction on OOM
            max_retries = 3
            current_batch_size = params.get('batch_size', 32)
            
            for retry in range(max_retries):
                try:
                    params['batch_size'] = current_batch_size
                    pred_map = self._run_inference(img_np, params)
                    break  # Success!
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and retry < max_retries - 1:
                        # OOM error - reduce batch size and retry
                        current_batch_size = max(1, current_batch_size // 2)
                        print(f"GPU OOM - Retrying with batch_size={current_batch_size}")
                        # Clear cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise  # Re-raise if not OOM or last retry

            # Post-process predictions
            pred_inst, inst_info_dict = self._post_process(
                pred_map,
                min_nucleus_size=params.get('min_nucleus_size', 10)
            )

            # Create overlay visualization
            overlay_image = self._create_overlay(
                img_np,
                pred_inst,
                inst_info_dict,
                draw_centroids=params.get('draw_centroids', True)
            )

            # Compile scores
            scores = self._compute_scores(inst_info_dict)

            # Create result string
            str_result = self._format_result_string(scores)

            # Convert back to PIL with RGBA mode
            result_pil = Image.fromarray(overlay_image.astype(np.uint8), mode='RGBA')

            return {
                'success': True,
                'processed_image': result_pil,
                'scores': scores,
                'str_result': str_result,
                'metadata': {
                    'model_mode': self.model_mode,
                    'patch_size': self.patch_input_shape,
                    'nr_types': self.nr_types,
                    'total_nuclei': scores.get('total_nuclei', 0)
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'str_result': f'Error: {str(e)}'
            }

    def _run_inference(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """
        Run HoVer-Net inference on image.

        Args:
            image: RGB image as numpy array
            params: Processing parameters

        Returns:
            Prediction map with channels [type_prob, nuclei_prob, h_dir, v_dir]
        """
        # Prepare patches
        patches, patch_info = self._prepare_patches(image)

        # Run model on patches
        pred_patches = []
        batch_size = params.get('batch_size', 32)
        
        # Clear GPU cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).float()
                batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
                batch_tensor = batch_tensor.to(self.device)

                # Forward pass
                pred_dict = self.model(batch_tensor)

                # Convert to NHWC
                pred_dict = OrderedDict([
                    [k, v.permute(0, 2, 3, 1).contiguous()]
                    for k, v in pred_dict.items()
                ])

                # Process predictions
                pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
                if "tp" in pred_dict:
                    type_map = F.softmax(pred_dict["tp"], dim=-1)
                    type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                    type_map = type_map.type(torch.float32)
                    pred_dict["tp"] = type_map

                # Concatenate predictions
                pred_output = torch.cat(list(pred_dict.values()), -1)
                pred_patches.append(pred_output.cpu().numpy())
                
                # Free GPU memory after each batch
                del batch_tensor, pred_dict, pred_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Concatenate all batches
        pred_patches = np.concatenate(pred_patches, axis=0)

        # Assemble patches back to full image
        pred_map = self._assemble_patches(
            pred_patches, patch_info, image.shape)

        return pred_map

    def _prepare_patches(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Prepare image patches for inference.

        Returns:
            patches: Array of patches (N, H, W, C)
            patch_info: Information for reassembling patches
        """
        h, w = image.shape[:2]
        input_size = self.patch_input_shape
        output_size = self.patch_output_shape
        step_size = output_size

        # Calculate padding
        pad_h = (step_size - h % step_size) % step_size
        pad_w = (step_size - w % step_size) % step_size

        # Pad image
        padded_image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='reflect'
        )

        # Extract patches
        patches = []
        patch_coords = []

        for y in range(0, padded_image.shape[0], step_size):
            for x in range(0, padded_image.shape[1], step_size):
                # Extract patch with margin
                margin = (input_size - output_size) // 2
                y_start = max(0, y - margin)
                y_end = min(padded_image.shape[0], y + step_size + margin)
                x_start = max(0, x - margin)
                x_end = min(padded_image.shape[1], x + step_size + margin)

                patch = padded_image[y_start:y_end, x_start:x_end]

                # Pad patch to input size if needed
                if patch.shape[0] < input_size or patch.shape[1] < input_size:
                    pad_y = input_size - patch.shape[0]
                    pad_x = input_size - patch.shape[1]
                    patch = np.pad(
                        patch,
                        ((0, pad_y), (0, pad_x), (0, 0)),
                        mode='reflect'
                    )

                patches.append(patch)
                patch_coords.append((y, x))

        patches = np.array(patches)
        patch_info = {
            'coords': patch_coords,
            'output_size': output_size,
            'padded_shape': padded_image.shape,
            'original_shape': (h, w)
        }

        return patches, patch_info

    def _assemble_patches(
        self,
        pred_patches: np.ndarray,
        patch_info: Dict,
        original_shape: Tuple
    ) -> np.ndarray:
        """Assemble prediction patches back to full image."""
        coords = patch_info['coords']
        output_size = patch_info['output_size']
        padded_shape = patch_info['padded_shape']

        # Create output array
        num_channels = pred_patches.shape[-1]
        pred_map = np.zeros(
            (padded_shape[0], padded_shape[1], num_channels),
            dtype=np.float32
        )

        # Place patches
        for idx, (y, x) in enumerate(coords):
            pred_patch = pred_patches[idx]
            # Extract center region
            margin = (pred_patch.shape[0] - output_size) // 2
            center_pred = pred_patch[margin:margin +
                                     output_size, margin:margin+output_size]

            pred_map[y:y+output_size, x:x+output_size] = center_pred

        # Crop to original size
        h, w = original_shape[:2]
        pred_map = pred_map[:h, :w]

        return pred_map

    def _post_process(
        self,
        pred_map: np.ndarray,
        min_nucleus_size: int = 10
    ) -> Tuple[np.ndarray, Dict]:
        """
        Post-process predictions to extract nucleus instances.

        Returns:
            pred_inst: Instance segmentation map
            inst_info_dict: Dictionary with info for each nucleus
        """
        from scipy import ndimage as ndi
        from skimage.morphology import remove_small_objects
        from skimage import measure

        # Extract channels
        if self.nr_types is not None:
            pred_type = pred_map[..., 0]
            pred_inst = pred_map[..., 1:]
        else:
            pred_inst = pred_map

        # Extract probability and horizontal/vertical gradient
        blb_raw = pred_inst[..., 0]
        h_dir_raw = pred_inst[..., 1]
        v_dir_raw = pred_inst[..., 2]

        # Threshold nuclei probability
        blb = (blb_raw >= 0.5).astype(np.int32)

        # Label connected components
        blb = measure.label(blb)
        blb = remove_small_objects(blb, min_size=min_nucleus_size)
        blb[blb > 0] = 1

        # Normalize direction maps
        h_dir = cv2.normalize(
            h_dir_raw, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        v_dir = cv2.normalize(
            v_dir_raw, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        # Combine with direction to separate touching nuclei
        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

        sobelh = 1 - (
            cv2.normalize(
                sobelh, None, alpha=0, beta=1,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv, None, alpha=0, beta=1,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        # Marker-controlled watershed
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        dist = ndi.distance_transform_edt(blb)
        # peak_local_max returns coordinates; create boolean mask from them
        local_max_coords = peak_local_max(
            dist, min_distance=7, threshold_abs=0,
            exclude_border=False
        )
        local_max = np.zeros(dist.shape, dtype=bool)
        if len(local_max_coords) > 0:
            local_max[local_max_coords[:, 0], local_max_coords[:, 1]] = True
        markers = measure.label(local_max)

        pred_inst = watershed(-overall, markers, mask=blb)

        # Extract instance information
        inst_info_dict = {}
        inst_id_list = np.unique(pred_inst)[1:]  # Exclude background

        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id

            # Get bounding box
            rmin, cmin, rmax, cmax = self._get_bounding_box(inst_map)

            # Get centroid
            inst_moment = cv2.moments(inst_map.astype(np.uint8))
            if inst_moment['m00'] != 0:
                cx = inst_moment['m10'] / inst_moment['m00']
                cy = inst_moment['m01'] / inst_moment['m00']
            else:
                cy, cx = ndi.center_of_mass(inst_map)

            # Get contour
            contours, _ = cv2.findContours(
                inst_map.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:
                contour = contours[0].squeeze()
                if contour.ndim == 1:
                    contour = contour.reshape(-1, 2)
            else:
                contour = np.array([[cmin, rmin], [cmax, rmax]])

            # Get type if available
            inst_type = None
            if self.nr_types is not None:
                inst_type_map = pred_type[inst_map]
                inst_type = int(np.median(inst_type_map))

            inst_info_dict[int(inst_id)] = {
                'bbox': np.array([[rmin, cmin], [rmax, cmax]]),
                'centroid': np.array([cx, cy]),
                'contour': contour,
                'type': inst_type,
                'area': np.sum(inst_map)
            }

        return pred_inst, inst_info_dict

    def _get_bounding_box(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of binary mask."""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return int(rmin), int(cmin), int(rmax + 1), int(cmax + 1)

    def _create_overlay(
        self,
        image: np.ndarray,
        inst_map: np.ndarray,
        inst_info_dict: Dict,
        draw_centroids: bool = True
    ) -> np.ndarray:
        """Create visualization overlay with transparent background and markers only."""
        # Create transparent RGBA overlay
        overlay = np.zeros((*inst_map.shape, 4), dtype=np.uint8)

        # Generate random colors for each instance
        np.random.seed(0)
        inst_ids = list(inst_info_dict.keys())
        colors = np.random.randint(50, 255, size=(len(inst_ids) + 1, 3))

        # Draw filled instances with transparency
        for idx, inst_id in enumerate(inst_ids, start=1):
            mask = inst_map == inst_id
            # Set RGBA color (R, G, B, A) with proper alpha
            r, g, b = colors[idx]
            overlay[mask, 0] = r
            overlay[mask, 1] = g
            overlay[mask, 2] = b
            overlay[mask, 3] = 200  # Alpha channel

        # Draw contours in RGB mode (OpenCV draws in BGR)
        for inst_id, info in inst_info_dict.items():
            contour = info['contour']
            if contour.shape[0] > 2:
                # OpenCV expects BGR, but we need RGB so swap
                cv2.drawContours(
                    overlay, [contour.astype(np.int32)], -1,
                    (0, 255, 255, 255), 2  # Yellow in BGR = (0,255,255)
                )

            # Draw centroid
            if draw_centroids:
                cx, cy = info['centroid']
                # Green in BGR = (0,255,0)
                cv2.circle(
                    overlay, (int(cx), int(cy)), 3,
                    (0, 255, 0, 255), -1
                )
        
        # Convert BGR to RGB (OpenCV uses BGR, PIL expects RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
        
        # save overlay for debugging
        cv2.imwrite("debug_overlay.png", overlay_rgb)
        # save overlay on original image for debugging
        blended = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.7,
            cv2.cvtColor(overlay_rgb[..., :3], cv2.COLOR_RGBA2BGR), 0.3, 0
        )
        cv2.imwrite("debug_blended.png", blended)
        return overlay_rgb

    def _compute_scores(self, inst_info_dict: Dict) -> Dict[str, Any]:
        """Compute quantitative scores from predictions."""
        scores = {
            'total_nuclei': len(inst_info_dict)
        }

        # Compute type-specific counts if available
        if self.nr_types is not None and self.type_info_dict:
            type_counts = {}
            for info in inst_info_dict.values():
                inst_type = info.get('type')
                if inst_type is not None:
                    type_name = self.type_info_dict.get(
                        inst_type, [f'Type {inst_type}', None]
                    )[0]
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1

            scores['nuclei_by_type'] = type_counts

        # Compute size statistics
        if inst_info_dict:
            areas = [info['area'] for info in inst_info_dict.values()]
            scores['mean_nucleus_area'] = float(np.mean(areas))
            scores['median_nucleus_area'] = float(np.median(areas))
            scores['std_nucleus_area'] = float(np.std(areas))

        return scores

    def _format_result_string(self, scores: Dict) -> str:
        """Format scores into display string."""
        total = scores.get('total_nuclei', 0)
        result_str = f"Detected {total} nuclei"

        if 'nuclei_by_type' in scores:
            type_counts = scores['nuclei_by_type']
            type_strs = [f"{count} {name}" for name,
                         count in type_counts.items()]
            result_str += " (" + ", ".join(type_strs) + ")"

        if 'mean_nucleus_area' in scores:
            mean_area = scores['mean_nucleus_area']
            result_str += f"\nMean nucleus area: {mean_area:.1f} px²"

        return result_str

    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """
        Define configurable parameters for frontend UI.

        Returns:
            Dictionary describing each hyperparameter with:
                - type: Parameter data type
                - default: Default value
                - min/max: Valid range (for numeric types)
                - choices: Valid values (for choice type)
                - description: Human-readable description
        """
        # Reload config from file to get fresh values
        current_model_configs = load_model_configs_from_file()
        
        # Build model variant choices with labels
        model_variant_choices = []
        for variant_id, variant_config in current_model_configs.items():
            model_variant_choices.append({
                'value': variant_id,
                'label': f"{variant_id.upper()} - {variant_config['description']}"
            })
        
        return {
            'model_variant': {
                'type': 'choice',
                'default': 'pannuke',
                'choices': model_variant_choices,
                'description': 'Select HoVer-Net model variant'
            },
            'batch_size': {
                'type': 'int',
                'default': 32,
                'min': 1,
                'max': 128,
                'description': 'Number of patches to process in parallel'
            },
            'draw_centroids': {
                'type': 'bool',
                'default': True,
                'description': 'Draw nucleus centroid points on overlay'
            },
            'min_nucleus_size': {
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 1000,
                'description': 'Minimum nucleus area in pixels'
            }
        }

    def cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        if self.model is not None:
            del self.model
            self.model = None

        self._is_initialized = False
        self.current_variant = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate_input(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input image and mask.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, "Image is required"

        # Check image size
        width, height = image.size
        if width < 64 or height < 64:
            return False, f"Image too small: {width}x{height}. Minimum 64x64."

        if width > 10000 or height > 10000:
            return False, f"Image too large: {width}x{height}. Maximum 10000x10000."

        # Check mask if provided
        if mask is not None:
            if mask.size != image.size:
                return False, (
                    f"Mask size {mask.size} does not match "
                    f"image size {image.size}"
                )

        return True, None

    def preprocess_image(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Apply preprocessing to input image.

        Args:
            image: Input image
            mask: Optional mask to apply

        Returns:
            Preprocessed image
        """
        if mask is None:
            return image

        # Apply mask (white out non-masked regions)
        img_np = np.array(image)
        mask_np = np.array(mask)

        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        img_np[mask_np == 0] = 255

        return Image.fromarray(img_np)

    def postprocess_output(
        self,
        output_image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Apply postprocessing to output image.

        Args:
            output_image: Model output image
            mask: Optional mask to apply

        Returns:
            Postprocessed image
        """
        if mask is None:
            return output_image

        # White out non-masked regions
        out_np = np.array(output_image)
        mask_np = np.array(mask)

        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        out_np[mask_np == 0] = 255

        return Image.fromarray(out_np)