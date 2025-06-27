from typing import Dict

import torch
import torch.nn as nn
from src.models.classifier.base_classifier import BaseClassifier

class DigitColorClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (3,28,28) → (32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                            # → (32,14,14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # → (64,7,7)
        )
        self.flatten = nn.Flatten()
        self.digit_head = nn.Linear(64 * 7 * 7, 10)
        self.color_head = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features = self.shared(x)
        features = features.flatten(1)
        return {
            'digit_logits': self.digit_head(features),
            'color_logits': self.color_head(features)
        }
