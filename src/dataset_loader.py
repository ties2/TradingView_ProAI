# src/dataset_loader.py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Use torchvision for common image transforms
from PIL import Image
import os
import pandas as pd # To load the CSV if you create one

class ChartDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {} # To map class names (folder names) to integers

        # Assuming data_dir is like 'dataset/train', 'dataset/val', 'dataset/test'
        # Each subdirectory is a class (e.g., 'uptrend', 'downtrend')
        for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Convert to RGB to handle different source image types
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
# Common practice to resize to a fixed size for CNNs (e.g., 224x224 or 256x256)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # All images will be resized to 224x224
    transforms.RandomHorizontalFlip(), # Data augmentation
    transforms.RandomRotation(10), # Data augmentation
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization (common)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For loading data in main.py:
# train_dataset = ChartDataset(data_dir='./dataset/train', transform=train_transforms)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# ... similar for val and test