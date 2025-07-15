import torch
import torch.nn as nn
from typing import Optional

class GuidedUNet(nn.Module):
    """Classifier-free guided UNet accepting multiple conditions."""
    def __init__(self, dim: int = 64, slots: int = 3, beta_schedule: str = "linear"):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(3 + slots, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, 3, 3, padding=1),
        )
        self.slots = slots
        self.beta_schedule = beta_schedule

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cond_all: torch.Tensor,
            cond_mask: Optional[torch.Tensor] = None,
    ):
        if cond_mask is None:
            cond_mask = torch.ones_like(cond_all)
            # cond_all and cond_mask may come as [B, slots]. Expand to match image dims
            if cond_all.ndim == 2:
                cond_all = cond_all.view(cond_all.size(0), cond_all.size(1), 1, 1)
            if cond_mask.ndim == 2:
                cond_mask = cond_mask.view(cond_mask.size(0), cond_mask.size(1), 1, 1)

            cond_all = cond_all.expand(-1, -1, x.size(2), x.size(3))
            cond_mask = cond_mask.expand_as(cond_all)
        x_in = torch.cat([x, cond_all * cond_mask], dim=1)
        return self.unet(x_in)