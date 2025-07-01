import torch
from torch import nn
from typing import List, Callable

class ItoSuperposition(nn.Module):
    """Combine multiple diffusion score networks using the Ito density estimator."""
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the summed score from each model."""
        scores = [m(x, t) for m in self.models]
        return torch.stack(scores, dim=0).sum(dim=0)

def and_sampler(score_models: List[nn.Module], sample_fn: Callable, shape: tuple,
                timesteps: int = 300, beta_start: float = 1e-4,
                beta_end: float = 2e-2) -> torch.Tensor:
    """Sample from the intersection distribution using superposed scores."""
    device = next(score_models[0].parameters()).device
    superposed = ItoSuperposition(score_models)
    def score_fn(x, t):
        return superposed(x, t)
    return sample_fn(score_fn, shape, timesteps=timesteps,
                     beta_start=beta_start, beta_end=beta_end, device=device)