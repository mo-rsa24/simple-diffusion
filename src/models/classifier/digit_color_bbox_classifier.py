# models/digit_color_bbox_classifier.py
from typing import Dict

import torch
from torch import nn

from src.models.classifier.base_classifier import BaseClassifier


class DigitColorBBoxClassifier(BaseClassifier):
    def __init__(self, in_channels=3, n_classes=10):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # (3,28,28) → (32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (32,14,14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (64,7,7)
        )
        self.flatten = nn.Flatten()
        self.digit_head = nn.Linear(64 * 7 * 7, n_classes)
        self.color_head = nn.Linear(64 * 7 * 7, n_classes)
        self.bbox_head = nn.Linear(64 * 7 * 7, n_classes)  # <-- ADD THIS

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features = self.shared(x)
        features = features.flatten(1)
        return {
            'digit_logits': self.digit_head(features),
            'color_logits': self.color_head(features),
            'bbox_logits':  self.bbox_head(features)
        }
