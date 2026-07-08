from torch import nn, optim
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from tqdm import tqdm

# Define the path to the input data
input_path = 'D:/Develop/UBC/Datasets/Run_155_ER//results_areas'
num_epochs = 10

# Define the transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset = datasets.ImageFolder(input_path, transform=transform)

# Get the labels for stratified splitting
labels = np.array(dataset.targets)

# Define the number of splits
n_splits = 5

# Create the stratified k-fold splitter
skf = StratifiedKFold(n_splits=n_splits)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {device}')

def create_model():
    # Define the model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    return model, criterion, optimizer

# Perform the stratified k-fold cross-test
accuracies, sensitivities, specificities = [], [], []
counter = 0
for train_index, test_index in skf.split(np.zeros(len(dataset)), labels):
    model, criterion, optimizer = create_model()
    # Create the train and test datasets
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    print(f'Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}')
    # Train the model
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        model.train()
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if(epoch % 1 == 0):
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

        if(epoch == num_epochs):
            torch.save(model.state_dict(), f'model_epoch_{epoch}_{counter}.pth')
            counter +=1
            # Evaluate the model
            model.eval()
            with torch.no_grad():
                correct, total, TP, TN, FP, FN = 0, 0, 0, 0, 0, 0
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    TP += ((predicted == 1) & (labels == 1)).sum().item()
                    TN += ((predicted == 0) & (labels == 0)).sum().item()
                    FP += ((predicted == 1) & (labels == 0)).sum().item()
                    FN += ((predicted == 0) & (labels == 1)).sum().item()
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            print("confusion matrix TP, TN, FP, FN:", TP, TN, FP, FN)
            print(f'Sensitivity: {sensitivity * 100}%')
            print(f'Specificity: {specificity * 100}%')
            print(f'Accuracy: {correct / total * 100}%')
            accuracies.append(correct / total * 100)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

print(f'Average accuracy: {np.mean(accuracies)}%')
print(f'Average sensitivity: {np.mean(sensitivities) * 100}%')
print(f'Average specificity: {np.mean(specificities) * 100}%')