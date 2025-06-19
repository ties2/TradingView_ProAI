# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns # For plotting data distribution

# Import custom modules
from src.dataset_loader import ChartDataset, train_transforms, val_test_transforms
from src.model_architecture import ChartPatternClassifier
from src.utils import evaluate_model, save_model, load_model

# --- 0. Device Configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS) for computations.")
else:
    device = torch.device("cpu")
    print("Apple Silicon GPU (MPS) not available or not an M1/M2/M3 Mac. Using CPU for computations.")


# --- 1. Data Preparation and Loading ---
# Adjust these paths to where your 'trading_chart_dataset' is located
TRAIN_DATA_DIR = './dataset/trading_chart_dataset/train'
VAL_DATA_DIR = './dataset/trading_chart_dataset/val'
TEST_DATA_DIR = './dataset/trading_chart_dataset/test'
MODEL_SAVE_PATH = './trained_model.pth'

# Create dataset instances
train_dataset = ChartDataset(data_dir=TRAIN_DATA_DIR, transform=train_transforms)
val_dataset = ChartDataset(data_dir=VAL_DATA_DIR, transform=val_test_transforms)
test_dataset = ChartDataset(data_dir=TEST_DATA_DIR, transform=val_test_transforms)

# Get class names and number of classes from the training dataset
class_names = list(train_dataset.class_to_idx.keys())
num_classes = len(class_names)
print(f"Detected classes: {class_names}")
print(f"Number of classes: {num_classes}")

# Create DataLoader instances
BATCH_SIZE = 16 # Adjust batch size based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# --- 2. Model Initialization ---
model = ChartPatternClassifier(num_classes=num_classes).to(device)

# --- 3. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. Training Loop ---
NUM_EPOCHS = 20 # You might need more epochs for better performance

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.4f}")

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)  "):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    val_loss = val_running_loss / len(val_dataset)
    val_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the trained model
save_model(model, MODEL_SAVE_PATH)

# --- 5. Evaluation on Test Set ---
print("\nEvaluating on test set...")
evaluate_model(model, test_loader, device, num_classes, class_names)

print("\nProject execution complete!")