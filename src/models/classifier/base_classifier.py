from typing import Dict

import torch
import torch.nn as nn

class BaseClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
