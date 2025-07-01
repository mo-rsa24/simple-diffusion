import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ClassifierGuidedUNet(nn.Module):
    """Simple UNet with an auxiliary classifier for guidance."""
    def __init__(self, dim: int = 64, channels: int = 3, num_classes: int = 10,
                 beta_schedule: str = "linear"):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(channels, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, channels, 3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(channels, dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )
        self.beta_schedule = beta_schedule

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        eps = self.unet(x)
        if self.training and labels is not None:
            logits = self.classifier(x)
            return eps, logits
        return eps