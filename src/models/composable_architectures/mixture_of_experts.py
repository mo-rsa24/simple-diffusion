import torch
import torch.nn as nn
from typing import List
import inspect

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
        eps_list = []
        for expert in self.experts:
            sig = inspect.signature(expert.forward)
            if len(sig.parameters) == 1:
                eps = expert(x)
            else:
                eps = expert(x, t)
            eps_list.append(eps)
        eps = torch.stack(eps_list, dim=1)
        alpha = self.mixer(x).unsqueeze(2)
        eps = eps * alpha
        return eps.sum(dim=1)