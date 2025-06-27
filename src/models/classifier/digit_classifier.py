import torch.nn as nn
from src.models.classifier.base_classifier import BaseClassifier

class DigitClassifier(BaseClassifier):
    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        self.backbone = SimpleCNN(in_channels, n_classes)

    def forward(self, x):
        return {'digit_logits': self.backbone(x)}

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(1, 16, 3, padding=1) where f(kernel)=3, p(padding)=1, s(stride)=1
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3 , padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = (nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 7 * 7, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=n_classes),
        ))

    def forward(self, x):
        return self.classifier(self.features(x) )