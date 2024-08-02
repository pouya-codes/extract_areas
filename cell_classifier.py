import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import cv2

class CellClassifier:
    def __init__(self, model_path, device=None, patch_size=64, batch_size=32, classifier_threshold=0.5, generate_gradcam=False):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.classifier_threshold = classifier_threshold
        self.generate_gradcam = generate_gradcam

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._create_model()
        self.gradient_cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
        self.gradient_cam.batch_size = self.batch_size
        self._load_model_weights()
        

    def _create_model(self):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        model = model.to(self.device)
        return model

    def _load_model_weights(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
    

    def predict_single(self, image):
        image = Image.fromarray(image)
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        predicted_label = predicted.item()
        predicted_probability = probabilities[0, predicted_label].item()
        
        return predicted_label, predicted_probability
    
    def _process_patches(self, patches, positions, heatmap, width, height):
        patches_tensor = torch.cat(patches)
        gradcams, labels, probabilities = self._generate_gradcam(patches_tensor)
        for i, (x_pos, y_pos) in enumerate(positions):
            gradcam_resized = np.array(Image.fromarray(gradcams[i]).resize((self.patch_size, self.patch_size), Image.BILINEAR))
            x_end = self.patch_size if (x_pos + self.patch_size) < width else width - x_pos
            y_end = self.patch_size if (y_pos + self.patch_size) < height else height - y_pos
            if labels[i] == 1:
                heatmap[y_pos:y_pos + y_end, x_pos:x_pos + x_end] += gradcam_resized[:y_end, :x_end]
        return labels, probabilities

    
    def _generate_gradcam(self, image_batch):
        with torch.no_grad():
            outputs = self.model(image_batch)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        predicted_labels = predicted.cpu().numpy()
        predicted_probabilities = probabilities[range(len(predicted)), predicted].cpu().numpy()

        target_category = [ClassifierOutputTarget(1)] * image_batch.shape[0] 
        cam = self.gradient_cam(input_tensor=image_batch, targets=target_category)
        return cam, predicted_labels, predicted_probabilities

    def process_image_with_sliding_window_batch(self, image, area):
        width, height = image.size
        heatmap = np.zeros((height, width))

        x, y, width, height, *path = area if len(area) == 5 else area + [None]
        if path is None:
            print("No path found for the area. Skipping...")
            return
        path = path[0]

        # subtract the x and y from the path vertices to covert global coordinates to local coordinates
        path.vertices -= np.array([x, y])

        patches = []
        positions = []
        voting = np.zeros(2)

        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                points = (x, y), (x + self.patch_size, y), (x + self.patch_size, y + self.patch_size), (x, y + self.patch_size)
                if np.all(path.contains_points(points)):
                    patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                    patch_tensor = self.transform(patch.convert('RGB')).unsqueeze(0).to(self.device)
                    patches.append(patch_tensor)
                    positions.append((x, y))

                    if len(patches) == self.batch_size:
                        labels, probabilities = self._process_patches(patches, positions, heatmap, width, height)
                        for idx, label in enumerate(labels):
                            if probabilities[idx] > self.classifier_threshold:
                                voting[label] += 1
                        patches = []
                        positions = []

        # Process remaining patches
        if patches:
            labels, probabilities = self._process_patches(patches, positions, heatmap, width, height)
            for idx, label in enumerate(labels):
                if probabilities[idx] > self.classifier_threshold:
                    voting[label] += 1

        overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        return Image.fromarray(overlay), self._create_json(voting)

    def _create_json(self, voting):
    # Create the JSON object
        num_total = int(voting.sum())
        num_pos = int(voting[1])
        num_neg = int(voting[0])
        percent_pos = round((num_pos / num_total) * 100, 1) if num_total > 0 else 0.0
        result = {
            "num_total": num_total,
            "num_pos": num_pos,
            "num_neg": num_neg,
            "percent_pos": percent_pos,
            "prob_thresh": self.classifier_threshold,
            "size_thresh": None,
            "size_thresh_upper": None,
            "marker_thresh": None,
            "cell_coords": []
        }
        return result
    
    def process_window(self, image, coords):
        coordinates_array = np.array(coords)
        # Calculate the mean along the vertical axis for this set of coordinates
        center = np.mean(coordinates_array, axis=0)
        y, x = center
        # Crop a 64x64 window around the (x, y) point
        half_size = self.patch_size//2
        y_min = int(max(0, y - half_size))
        y_max = int(min(image.shape[0], y + half_size))
        x_min = int(max(0, x - half_size))
        x_max = int(min(image.shape[1], x + half_size))
        
        cropped_image = image[y_min:y_max, x_min:x_max]

        # # Ensure the cropped image is 64x64
        # if cropped_image.shape[0] != 64 or cropped_image.shape[1] != 64:
        #     padded_image = np.zeros((64, 64, 3), dtype=np.uint8)
        #     padded_image[:cropped_image.shape[0], :cropped_image.shape[1], :] = cropped_image
        #     cropped_image = padded_image

        return self.predict_single(cropped_image)
