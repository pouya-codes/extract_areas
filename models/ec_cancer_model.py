"""
EC Cancer Classification Model

This model implements endometrial cancer (EC) subtype classification using a
VarMIL (Variance-based Multiple Instance Learning) approach. It processes
whole slide images or regions to classify between NSMP and p53 subtypes.

The model uses a three-stage pipeline:
1. Tissue mask generation with SAM2
2. Patch-level tumor classification with ResNet50
3. Slide-level aggregation with VarMIL
"""

from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import sys
from matplotlib.path import Path as MplPath

from models.base_model import BaseAIModel


# ============================================================================
# Embedded Model Classes from EC_Pipeline
# ============================================================================

# Model output channel sizes for different backbones
out_channel = {
    'alexnet': 256, 'vgg16': 512, 'vgg19': 512, 'vgg16_bn': 512, 'vgg19_bn': 512,
    'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnext50_32x4d': 2048,
    'resnext101_32x8d': 2048, 'mobilenet_v2': 1280, 'mobilenet_v3_small': 576,
    'mobilenet_v3_large': 960, 'mnasnet1_3': 1280, 'shufflenet_v2_x1_5': 1024,
    'squeezenet1_1': 512, 'efficientnet-b0': 1280, 'efficientnet-l2': 5504,
    'efficientnet-b1': 1280, 'efficientnet-b2': 1408, 'efficientnet-b3': 1536,
    'efficientnet-b4': 1792, 'efficientnet-b5': 2048, 'efficientnet-b6': 2304,
    'efficientnet-b7': 2560, 'efficientnet-b8': 2816
}

feature_map = {
    'alexnet': -2, 'vgg16': -2, 'vgg19': -2, 'vgg16_bn': -2, 'vgg19_bn': -2,
    'resnet18': -2, 'resnet34': -2, 'resnet50': -2, 'resnext50_32x4d': -2,
    'resnext101_32x8d': -2, 'mobilenet_v2': 0, 'mobilenet_v3_large': -2,
    'mobilenet_v3_small': -2, 'mnasnet1_3': 0, 'shufflenet_v2_x1_5': -1,
    'squeezenet1_1': 0
}

diff_fc_layer = ['mobilenet_v2', 'mnasnet1_3', 'shufflenet_v2_x1_5']


class Model(nn.Module):
    """
    Patch classifier model wrapper.
    Simplified version from submodule_cv.deep_models.models.
    """
    def __init__(self, config):
        super().__init__()
        self.base_model = config["base_model"]
        self.num_classes = config["num_subtypes"]
        self.pretrained = config["pretrained"]
        
        # Load base model
        model = getattr(models, self.base_model)
        model = model(pretrained=self.pretrained)
        
        # Modify last layer for classification
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, self.num_classes)
        
        # Separate feature extractor and classifier
        self.feature_extract = nn.Sequential(
            *list(model.children())[:feature_map[self.base_model]]
        )
        self.classifier = nn.Sequential(
            *list(model.children())[feature_map[self.base_model]:]
        )
    
    def forward(self, x):
        feature = self.feature_extract(x)
        feature_pool = self.classifier[0](feature)
        flatten_feature = torch.flatten(feature_pool, 1)
        out = self.classifier[1:](flatten_feature)
        return out

feature_map = {
    'alexnet': -2, 'vgg16': -2, 'vgg19': -2, 'vgg16_bn': -2, 'vgg19_bn': -2,
    'resnet18': -2, 'resnet34': -2, 'resnet50': -2, 'resnext50_32x4d': -2,
    'resnext101_32x8d': -2, 'mobilenet_v2': 0, 'mobilenet_v3_large': -2,
    'mobilenet_v3_small': -2, 'mnasnet1_3': 0, 'shufflenet_v2_x1_5': -1,
    'squeezenet1_1': 0
}


class VanillaModel(nn.Module):
    """Feature extraction model using various CNN backbones."""
    
    def __init__(self, backbone):
        super(VanillaModel, self).__init__()
        self.backbone = backbone
        model = getattr(models, self.backbone)
        model = model(pretrained=False)
        # Separate feature and classifier layers
        if feature_map[self.backbone] == 0:
            self.feature_extract = nn.Sequential(*list(model.children())[0])
        else:
            self.feature_extract = nn.Sequential(
                *list(model.children())[:feature_map[self.backbone]]
            )

    def forward(self, x):
        feature = self.feature_extract(x)
        feature = F.adaptive_avg_pool2d(feature, 1)
        out = torch.flatten(feature, 1)
        return out


class VarMIL(nn.Module):
    """Variance-based Multiple Instance Learning model."""
    
    def __init__(self, backbone, num_classes):
        super().__init__()
        dim = 128
        torch.autograd.set_detect_anomaly(True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention = nn.Sequential(
            nn.Linear(out_channel[backbone], dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * out_channel[backbone], dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """
        x   (input)            : B (batch size) x K (nb_patch) x out_channel
        A   (attention weights): B (batch size) x K (nb_patch) x 1
        M   (weighted mean)    : B (batch size) x out_channel
        S   (std)              : B (batch size) x K (nb_patch) x out_channel
        V   (weighted variance): B (batch size) x out_channel
        nb_patch (nb of patch) : B (batch size)
        M_V (concate M and V)  : B (batch size) x 2*out_channel
        out (final output)     : B (batch size) x num_classes
        """
        b, k, c = x.shape
        A = self.attention(x)
        # Filter padded rows
        A = A.masked_fill((x == 0).all(dim=2).reshape(A.shape), -9e15)
        A = F.softmax(A, dim=1)  # softmax over K
        M = torch.einsum('b k d, b k o -> b o', A, x)  # d is 1 here
        S = torch.pow(x - M.reshape(b, 1, c), 2)
        V = torch.einsum('b k d, b k o -> b o', A, S)
        nb_patch = (torch.tensor(k).expand(b)).to(self.device)
        # Filter padded rows
        nb_patch = nb_patch - torch.sum((x == 0).all(dim=2), dim=1)
        nb_patch = nb_patch / (nb_patch - 1)  # I / I-1
        # For cases when we have only 1 patch (inf)
        nb_patch = torch.nan_to_num(nb_patch, posinf=1)
        V = V * nb_patch[:, None]  # broadcasting
        M_V = torch.cat((M, V), dim=1)
        out = self.classifier(M_V)
        return A, out


# ============================================================================
# Custom Module Remapper for Pickle Loading
# ============================================================================

class ModuleRemapper:
    """
    Helper class to remap module paths during unpickling.
    This allows loading models that were saved with different module paths.
    """
    @staticmethod
    def remap_storage(storage, location):
        return storage
    
    @staticmethod
    def setup_module_aliases():
        """
        Create module aliases to handle models saved with submodule_cv.
        """
        # Create a fake submodule_cv package structure
        if 'submodule_cv' not in sys.modules:
            import types
            
            # Create the package hierarchy
            submodule_cv = types.ModuleType('submodule_cv')
            submodule_cv.__package__ = 'submodule_cv'
            submodule_cv.__path__ = []  # Make it a package
            
            submodule_cv_models = types.ModuleType('submodule_cv.models')
            submodule_cv_models.__package__ = 'submodule_cv.models'
            
            submodule_cv_deep_models = types.ModuleType(
                'submodule_cv.deep_models'
            )
            submodule_cv_deep_models.__package__ = 'submodule_cv.deep_models'
            submodule_cv_deep_models.__path__ = []  # Make it a package
            
            # Add submodule_cv.deep_models.models
            submodule_cv_deep_models_models = types.ModuleType(
                'submodule_cv.deep_models.models'
            )
            submodule_cv_deep_models_models.__package__ = (
                'submodule_cv.deep_models.models'
            )
            
            # Add our embedded models to all the fake modules
            submodule_cv_models.VanillaModel = VanillaModel
            submodule_cv_models.DeepModel = VanillaModel
            submodule_cv_deep_models.VanillaModel = VanillaModel
            submodule_cv_deep_models.DeepModel = VanillaModel
            submodule_cv_deep_models_models.VanillaModel = VanillaModel
            submodule_cv_deep_models_models.DeepModel = VanillaModel
            submodule_cv_deep_models_models.Model = Model
            
            # Register all modules
            sys.modules['submodule_cv'] = submodule_cv
            sys.modules['submodule_cv.models'] = submodule_cv_models
            sys.modules['submodule_cv.deep_models'] = submodule_cv_deep_models
            sys.modules['submodule_cv.deep_models.models'] = (
                submodule_cv_deep_models_models
            )
            
            # Link them as attributes
            submodule_cv.models = submodule_cv_models
            submodule_cv.deep_models = submodule_cv_deep_models
            submodule_cv_deep_models.models = submodule_cv_deep_models_models


# ============================================================================


class ECCancerModel(BaseAIModel):
    """
    Endometrial Cancer Classification Model.
    
    This model classifies endometrial cancer subtypes (NSMP vs p53) using a
    multi-stage deep learning pipeline that analyzes tissue patches and
    aggregates predictions at the slide/region level.
    
    Key Features:
    - Automatic tissue segmentation with SAM2
    - Patch-level tumor detection with ResNet50
    - Slide-level classification with VarMIL
    - NSMP vs p53 subtype prediction
    """
    
    def __init__(self):
        super().__init__(
            model_name="ec_cancer",
            model_version="1.0.0"
        )
        self.description = (
            "Endometrial Cancer subtype classifier using VarMIL. "
            "Classifies tissue regions as NSMP or p53 molecular subtypes "
            "through patch-based analysis and multiple instance learning."
        )
        self.requires_mask = True  # Requires mask or annotation points
        self.supported_input_formats = ['PNG', 'JPEG', 'JPG', 'TIFF', 'TIF']
        
        # Model components
        self.patch_classifier = None
        self.representation_generator = None
        self.varmil_model = None
        self.device = None
        self.transform = None
        
        # Default hyperparameters
        self.hyperparameters = {
            'patch_size': 1024,
            'resize_size': 512,
            'stride': 1,
            'batch_size': 32,
            'tumor_threshold': 0.9,
            'generate_visualization': True
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the EC Cancer model with all required components.
        
        Args:
            config: Dictionary with keys:
                - patch_classifier_model_path: Path to tumor classifier weights
                - representation_generator_model_path: Path to feature extractor weights
                - varmil_model_path: Path to VarMIL model weights
                - device: 'cuda' or 'cpu' (default: auto-detect)
        
        Raises:
            ValueError: If required paths are missing
            RuntimeError: If model loading fails
        """
        # Check required paths
        required_paths = [
            'patch_classifier_model_path',
            'representation_generator_model_path',
            'varmil_model_path'
        ]
        
        for path_key in required_paths:
            if path_key not in config:
                raise ValueError(f"Config must include '{path_key}'")
            if not Path(config[path_key]).exists():
                raise ValueError(f"Model file not found: {config[path_key]}")
        
        # Set device
        device_config = config.get('device', 'auto')
        if device_config == 'auto':
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device_str = device_config
        
        self.device = torch.device(device_str)
        print(f"EC Cancer model using device: {self.device}")
        
        # Setup module aliases for unpickling models with submodule_cv references
        ModuleRemapper.setup_module_aliases()
        
        # Initialize patch classifier (ResNet50)
        try:
            model = torch.load(
                config['patch_classifier_model_path'],
                map_location=self.device,
                weights_only=False
            )
            self.patch_classifier = model.model.to(self.device)
            self.patch_classifier.eval()
            
            # Define preprocessing transforms
            resize_size = config.get('resize_size', 512)
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.7784, 0.7081, 0.7951],
                    std=[0.1685, 0.2008, 0.1439]
                ),
            ])
            print("✓ Loaded patch classifier")
        except Exception as e:
            raise RuntimeError(f"Failed to load patch classifier: {e}")
        
        # Initialize representation generator
        try:
            state = torch.load(
                config['representation_generator_model_path'],
                map_location=self.device,
                weights_only=False
            )['model']
            model = VanillaModel("resnet34")
            model.eval()
            self.representation_generator = model.to(self.device)
            _, error_keys = self.representation_generator.load_state_dict(
                state, strict=False
            )
            # Verify only classifier keys are missing (expected)
            for key in error_keys:
                if "classifier" not in key:
                    raise ValueError(f"Unexpected missing key: {key}")
            print("✓ Loaded representation generator")
        except Exception as e:
            raise RuntimeError(f"Failed to load representation generator: {e}")
        
        # Initialize VarMIL model
        try:
            state = torch.load(
                config['varmil_model_path'],
                map_location=self.device,
                weights_only=False
            )
            model = VarMIL("resnet34", 2)
            state_dict = state['model']
            # Remove 'model.' prefix from keys
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "")
                new_state_dict[new_key] = v
            model.eval()
            self.varmil_model = model.to(self.device)
            self.varmil_model.load_state_dict(new_state_dict, strict=True)
            print("✓ Loaded VarMIL model")
        except Exception as e:
            raise RuntimeError(f"Failed to load VarMIL model: {e}")
        
        self._is_initialized = True
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a tissue region to classify EC subtype.
        
        Args:
            image: Input tissue region (RGB PIL Image)
            mask: Optional binary mask (if None, will be generated)
            annotation_points: Optional polygon points defining region
            hyperparameters: Optional dict with:
                - patch_size: Size of patches to extract (default: 1024)
                - resize_size: Size to resize patches before classification (default: 512)
                - stride: Stride for sliding window (default: 1)
                - batch_size: Batch size for processing (default: 32)
                - tumor_threshold: Confidence threshold for tumor patches (default: 0.9)
                - generate_visualization: Whether to create visualization (default: True)
        
        Returns:
            Dictionary containing:
                - processed_image: Visualization showing tumor regions
                - scores: Classification probabilities and metrics
                - metadata: Processing information
                - success: True if successful
                - error: Error message if failed
        """
        import time
        start_time = time.time()
        
        try:
            # Merge hyperparameters
            params = self.hyperparameters.copy()
            if hyperparameters:
                params.update(hyperparameters)
            
            # Validate input
            is_valid, error_msg = self.validate_input(image, mask)
            if not is_valid:
                return {
                    'success': False,
                    'error': error_msg,
                    'processed_image': None,
                    'scores': {}
                }
            
            # Generate mask from annotation points or use provided mask
            if mask is None and annotation_points is None:
                return {
                    'success': False,
                    'error': 'Either mask or annotation_points must be provided',
                    'processed_image': None,
                    'scores': {}
                }
            elif annotation_points is not None:
                # Create mask from annotation points
                mask = Image.new('L', image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.polygon(annotation_points, fill=255)
            
            # Extract tumor patches and generate representations
            representations = self._extract_tumor_representations(
                image,
                mask,
                params
            )
            
            if len(representations) == 0:
                return {
                    'success': True,
                    'processed_image': image,
                    'scores': {
                        'nsmp_probability': 0.0,
                        'p53_probability': 0.0,
                        'tumor_patches_found': 0,
                        'classification': 'insufficient_tissue'
                    },
                    'metadata': {
                        'processing_time': time.time() - start_time,
                        'message': 'No tumor patches detected in region'
                    }
                }
            
            # Create bag for VarMIL
            bag = torch.stack(representations).unsqueeze(0)
            
            # Run VarMIL classification
            with torch.no_grad():
                _, output = self.varmil_model.forward(bag)
                probs = torch.softmax(output, dim=1)
                nsmp_prob = float(probs[0][0].cpu().item())
                p53_prob = float(probs[0][1].cpu().item())
            
            # Determine classification
            classification = "NSMP" if nsmp_prob > p53_prob else "p53"
            confidence = max(nsmp_prob, p53_prob)
            
            # Generate visualization if requested
            if params['generate_visualization']:
                visualization = self._create_visualization(
                    image,
                    mask,
                    classification,
                    confidence
                )
            else:
                visualization = image
            
            return {
                'success': True,
                'processed_image': visualization,
                'scores': {
                    'nsmp_probability': round(nsmp_prob, 4),
                    'p53_probability': round(p53_prob, 4),
                    'classification': classification,
                    'confidence': round(confidence, 4),
                    'tumor_patches_found': len(representations)
                },
                'metadata': {
                    'processing_time': round(time.time() - start_time, 2),
                    'model_version': self.model_version,
                    'hyperparameters': params
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}",
                'processed_image': None,
                'scores': {}
            }
    
    def _extract_tumor_representations(
        self,
        image: Image.Image,
        mask: Image.Image,
        params: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """
        Extract feature representations from tumor patches.
        
        Args:
            image: Input image
            mask: Binary mask
            params: Processing parameters
        
        Returns:
            List of feature tensors
        """
        representations = []
        
        # Convert mask to numpy for processing
        mask_array = np.array(mask)
        
        # Create polygon path from mask
        contours, _ = cv2.findContours(
            mask_array,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return representations
        
        # Process each contour
        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < 100:
                continue
            
            # Create matplotlib path for point checking
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue
            path = MplPath(points)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Generate sliding window positions
            patch_size = params['patch_size']
            stride = params['stride']
            
            x_range = np.arange(x, x + w, patch_size * stride)
            y_range = np.arange(y, y + h, patch_size * stride)
            grid_x, grid_y = np.meshgrid(x_range, y_range)
            grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
            
            # Check which points are inside the contour
            inside_points = path.contains_points(grid_points)
            
            # Extract and classify patches
            patches = []
            for point, inside in zip(grid_points, inside_points):
                if not inside:
                    continue
                
                px, py = point
                # Ensure we don't go out of bounds
                if px + patch_size > image.width or py + patch_size > image.height:
                    continue
                
                patch = image.crop((px, py, px + patch_size, py + patch_size))
                patches.append(self.transform(patch))
            
            # Process patches in batches
            batch_size = params['batch_size']
            tumor_threshold = params['tumor_threshold']
            
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i + batch_size]
                
                if not batch_patches:
                    continue
                
                batch_tensor = torch.stack(batch_patches).to(self.device)
                
                with torch.no_grad():
                    # Classify patches
                    outputs = self.patch_classifier(batch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_probs = probs.cpu().numpy()
                    labels = np.argmax(pred_probs, axis=1)
                    
                    # Identify tumor patches
                    tumor_positive = (
                        (np.max(pred_probs, axis=1) > tumor_threshold) &
                        (labels == 1)
                    )
                    
                    # Extract tumor patches
                    tumor_patches = [
                        patch for patch, is_tumor in 
                        zip(batch_tensor, tumor_positive) if is_tumor
                    ]
                    
                    # Generate representations for tumor patches
                    if tumor_patches:
                        tumor_batch = torch.stack(tumor_patches)
                        reps = self.representation_generator(tumor_batch)
                        representations.extend(reps)
        
        return representations
    
    def _create_visualization(
        self,
        image: Image.Image,
        mask: Image.Image,
        classification: str,
        confidence: float
    ) -> Image.Image:
        """
        Create visualization with classification overlay.
        
        Args:
            image: Original image
            mask: Tissue mask
            classification: Classification result
            confidence: Confidence score
        
        Returns:
            Visualization image
        """
        # Create a copy for visualization
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image, 'RGBA')
        
        # Add semi-transparent overlay based on classification
        mask_array = np.array(mask)
        contours, _ = cv2.findContours(
            mask_array,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Color based on classification
        if classification == "p53":
            color = (255, 0, 0, 80)  # Red with transparency
        else:
            color = (0, 255, 0, 80)  # Green with transparency
        
        # Draw contours
        for contour in contours:
            points = contour.squeeze()
            if len(points.shape) == 1 or len(points) < 3:
                continue
            points = [(int(p[0]), int(p[1])) for p in points]
            draw.polygon(points, fill=color, outline=color[:3] + (255,))
        
        # Add text label
        label_text = f"{classification} ({confidence:.2%})"
        # Draw text with background
        text_bbox = draw.textbbox((10, 10), label_text)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
        draw.text((10, 10), label_text, fill=(255, 255, 255, 255))
        
        return vis_image
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """
        Get schema for EC Cancer model hyperparameters.
        
        Returns:
            Dictionary defining available hyperparameters
        """
        return {
            'patch_size': {
                'type': 'int',
                'default': 1024,
                'min': 256,
                'max': 2048,
                'description': 'Size of patches to extract from tissue (pixels)'
            },
            'resize_size': {
                'type': 'int',
                'default': 512,
                'min': 128,
                'max': 1024,
                'description': 'Size to resize patches before classification'
            },
            'stride': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 4,
                'description': 'Stride multiplier for sliding window (higher = faster but less coverage)'
            },
            'batch_size': {
                'type': 'int',
                'default': 32,
                'min': 1,
                'max': 128,
                'description': 'Number of patches to process in parallel'
            },
            'tumor_threshold': {
                'type': 'float',
                'default': 0.9,
                'min': 0.5,
                'max': 1.0,
                'description': 'Confidence threshold for tumor patch detection'
            },
            'generate_visualization': {
                'type': 'bool',
                'default': True,
                'description': 'Generate visualization with classification overlay'
            }
        }
    
    def cleanup(self) -> None:
        """
        Clean up GPU memory and resources.
        """
        if self.patch_classifier is not None:
            del self.patch_classifier
        if self.representation_generator is not None:
            del self.representation_generator
        if self.varmil_model is not None:
            del self.varmil_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("EC Cancer model cleaned up")
