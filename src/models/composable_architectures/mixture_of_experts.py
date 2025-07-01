import torch
import torch.nn as nn
from typing import List

class MixtureOfExperts(nn.Module):
    """Combine several expert score networks with mixing coefficients."""
    def __init__(self, experts: List[nn.Module], dim: int = 64, beta_schedule: str = "linear"):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.mixer = nn.Sequential(
            nn.Conv2d(3, len(experts), 3, padding=1),
            nn.Softmax(dim=1),
        )
        self.beta_schedule = beta_schedule

    def forward(self, x: torch.Tensor, t: torch.Tensor, masks: List[torch.Tensor]):
        eps = torch.stack([expert(x, t) for expert in self.experts], dim=1)
        alpha = self.mixer(x).unsqueeze(2)
        eps = eps * alpha
        return eps.sum(dim=1)