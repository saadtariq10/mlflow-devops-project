import os
import mlflow
import mlflow.pytorch
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchvision.models as models

# Create MLflow experiment if it doesn't exist
experiment_name = "CatDogClassifier"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f"Error setting up MLflow experiment: {e}")

class PetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Reshape considering RGB channels (3, 64, 64)
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            try:
                img = Image.open(filepath).convert('RGB')  # Convert to RGB
                img = img.resize((64, 64))
                img_array = np.array(img)  # Shape will be (64, 64, 3)
                images.append(img_array)
                labels.append(1 if label == 'dogs' else 0)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return np.array(images), np.array(labels)

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Just normalization for validation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading and preprocessing data...")
# Load and preprocess training data
X, y = load_images_from_folder("dataset/training_set")
print(f"Loaded {len(X)} images with shape {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets
train_dataset = PetDataset(X_train, y_train, transform=train_transform)
val_dataset = PetDataset(X_val, y_val, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with mlflow.start_run():
    # Load pre-trained ResNet18 and modify for binary classification
    model = models.resnet18(weights='IMAGENET1K_V1')  # Updated from pretrained=True
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    best_acc = 0
    num_epochs = 10

    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            # Handle single-sample case
            if len(outputs.shape) == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                # Handle single-sample case
                if len(outputs.shape) == 0:
                    outputs = outputs.unsqueeze(0)
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels.float()).item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        
        # Log metrics for this epoch
        mlflow.log_metric("train_loss", running_loss/len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save best model
            mlflow.pytorch.log_model(
                model,
                artifact_path="best_model",
                registered_model_name="CatDogModel"
            )

    # Log parameters and metrics
    mlflow.log_param("model", "ResNet18")
    mlflow.log_param("image_size", "64x64")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", num_epochs)
    mlflow.log_metric("best_val_accuracy", best_acc)

    print(f"Best Validation Accuracy: {best_acc:.4f}")
