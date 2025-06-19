# src/model_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models # For using pre-trained models

class ChartPatternClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ChartPatternClassifier, self).__init__()
        # Using a pre-trained ResNet as a backbone (transfer learning)
        # ResNet18 is a good starting point for smaller datasets
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze backbone layers if desired (optional for small datasets)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Replace the final classification layer
        # The number of input features to this layer depends on the backbone (e.g., 512 for ResNet18)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)