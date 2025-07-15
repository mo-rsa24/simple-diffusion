import torch
import torch.nn as nn
from typing import List
import inspect

class CascadedDiffusion(nn.Module):
    """Run a sequence of conditional diffusion models."""
    def __init__(self, stages: List[nn.Module], beta_schedule: str = "linear"):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.beta_schedule = beta_schedule

    def forward(self, x: torch.Tensor, t: torch.Tensor, factors: List[torch.Tensor]):
        out = x
        for stage, cond in zip(self.stages, factors):
            sig = inspect.signature(stage.forward)
            num_args = len(sig.parameters)
            if num_args == 1:
                out = stage(out)
            elif num_args == 2:
                out = stage(out, t)
            else:
                out = stage(out, t, cond)
        return out