import torch
import torch.nn as nn
from typing import List, Optional

class FactorNet(nn.Module):
    def __init__(self, dim: int = 64, channels: int = 3):
        super().__init__()
        # placeholder UNet block
        self.net = nn.Sequential(
            nn.Conv2d(channels, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ignore cond for skeleton
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, factors: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(factors, factors),
            nn.Sigmoid(),
        )

    def forward(self, scores: List[torch.Tensor]) -> torch.Tensor:
        # scores: list of tensors [B,C,H,W]
        stacked = torch.stack(scores, dim=1)
        b, f, c, h, w = stacked.shape
        gate = stacked.mean(dim=[2, 3, 4])  # [B, f]
        gate = self.fc(gate)  # [B, f]
        gate = gate.view(b, f, 1, 1, 1)
        return (gate * stacked).sum(dim=1)

class CompositionalUNet(nn.Module):
    """Factorized UNet with learnable gating."""

    def __init__(
            self,
            dim: int = 64,
            factors: int = 3,
            beta_schedule: str = "linear",
            use_attention_at: Optional[List[int]] = None,
    ):
        super().__init__()
        self.factors = nn.ModuleList([FactorNet(dim=dim) for _ in range(factors)])
        self.gating = GatingNetwork(factors)
        self.beta_schedule = beta_schedule
        self.use_attention_at = use_attention_at or []

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditions: List[torch.Tensor]):
        scores = [net(x, t, cond) for net, cond in zip(self.factors, conditions)]
        return self.gating(scores)