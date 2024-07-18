import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F

class CellClassifier:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._create_model()
        self._load_model_weights()

    def _create_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        model = model.to(self.device)
        return model

    def _load_model_weights(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def predict(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()
    
    def predict_batch(self, patches):
        patches = torch.from_numpy(patches).float()
        patches = torch.stack([self.transform(Image.fromarray(patch)) for patch in patches])
        patches = patches.to(self.device)

        with torch.no_grad():
            outputs = self.model(patches)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()
    

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
    
    
    def process_window(self, image, coords):
        coordinates_array = np.array(coords)
        # Calculate the mean along the vertical axis for this set of coordinates
        center = np.mean(coordinates_array, axis=0)
        y, x = center
        # Crop a 64x64 window around the (x, y) point
        half_size = 32
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
