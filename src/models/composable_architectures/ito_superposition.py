import torch
from typing import List, Callable
from src.models.superposition.ito_operator import ItoSuperposition


def superposed_sample(models: List[Callable], sampler: Callable, shape: tuple, beta_schedule: str = "linear", timesteps: int = 300):
    superposed = ItoSuperposition(models)
    def score_fn(x, t):
        return superposed(x, t)
    return sampler(score_fn, shape, timesteps=timesteps)
