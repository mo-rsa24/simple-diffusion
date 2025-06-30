from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import autocast

# Re‑use the existing backbone and embeddings
from src.models.vanilla.unet import Unet  # noqa: E402

# -----------------------------------------------------------------------------
# 𝟙.  SDE Utilities
# -----------------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────
#  Minimal VP-SDE helper
# ──────────────────────────────────────────────────────────────
class VPSDE:
    """
    Variance-Preserving SDE used in Song et al. (2021).  Parameterisation:
        dX = -½β(t) X dt + √β(t) dW ,    t ∈ [0, 1]
    We use the cosine schedule from Nicoli et al. (2022) by default.
    """
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min, self.beta_max = beta_min, beta_max

    # continuous beta(t)
    def beta(self, t: torch.Tensor):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    # closed-form log-SNR used by VP-SDE
    def marginal_std(self, t: torch.Tensor):
        c = torch.exp(-0.5 * (self.beta_min * t + 0.5 *
                              (self.beta_max - self.beta_min) * t ** 2))
        return torch.sqrt(1. - c ** 2)

    def marginal_mean(self, x0: torch.Tensor, t: torch.Tensor):
        c = torch.exp(-0.5 * (self.beta_min * t + 0.5 *
                              (self.beta_max - self.beta_min) * t ** 2))
        return c[:, None, None, None] * x0

    def marginal_sample(self, x0: torch.Tensor, t: torch.Tensor):
        std  = self.marginal_std(t)
        mean = self.marginal_mean(x0, t)
        noise = torch.randn_like(x0)
        return mean + std[:, None, None, None] * noise, noise, std
# -----------------------------------------------------------------------------
# 𝟚.  Score Network Wrapper
# -----------------------------------------------------------------------------

class ScoreSdeUNet(nn.Module):
    """Wraps the existing UNet so that it outputs ∇ₓ log pₜ(x)."""

    def __init__(self, sde: VPSDE | None = None, **unet_kwargs):
        super().__init__()
        self.unet = Unet(**unet_kwargs)
        self.sde = sde if sde is not None else VPSDE()

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return *scaled* score so that loss = ||σₜ s_θ(xₜ,t) + ε||²."""
        # Ensure t ∈ [0,1]
        if t.dtype.is_floating_point:
            t_in = t
        else:  # compatibility with integer timesteps
            t_in = t.float() / (t.max() + 1e-8)

        # Unet expects a (B,) vector of timesteps
        out = self.unet(x, t_in)
        sigma = self.sde.marginal_std(t_in)[:, None, None, None]
        return out / sigma  # scale as in Song (makes target noise N(0,1))

# ──────────────────────────────────────────────────────────────
#  Denoising-Score-Matching loss
# ──────────────────────────────────────────────────────────────
def dsm_loss(model, x0, t, sde: VPSDE):
    """
    Implements   λ(t) · || sθ(x_t, t) + ε/σ(t) ||²
    where λ(t)=σ(t)²   (Song et al., 2020), ε ~ N(0, I).
    """
    x_t, noise, std = sde.marginal_sample(x0, t)
    with autocast():
        score_pred = model(x_t, t)              # predicts ∂ₓ log p(x_t)
    target = -noise / std[:, None, None, None]  # analytical score
    weight = std ** 2                          # λ(t)
    loss = 0.5 * (weight[:, None, None, None] *
                  (score_pred - target).pow(2)).mean()
    return loss

# ──────────────────────────────────────────────────────────────
#  Predictor–Corrector sampler (simplified)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def pc_sampler(model, sde: VPSDE, shape, n_steps=100, predictor_steps=1,
               corrector_steps=1):
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)

    timesteps = torch.linspace(1., 1e-3, n_steps, device=device)

    for i in range(n_steps):
        t = timesteps[i].expand(shape[0]).to(device)

        # ── Corrector (Langevin dynamics) ───────────────
        for _ in range(corrector_steps):
            grad = model(x, t)
            noise = torch.randn_like(x)
            step_size = (sde.marginal_std(t) ** 2) * 0.1
            x = x + step_size[:, None, None, None] * grad + \
                torch.sqrt(2. * step_size)[:, None, None, None] * noise

        # ── Predictor (reverse SDE Euler) ───────────────
        dt = timesteps[i] - timesteps[i + 1] if i < n_steps - 1 else timesteps[i]
        drift = -0.5 * sde.beta(t)[:, None, None, None] * x
        diffusion = torch.sqrt(sde.beta(t))[:, None, None, None]
        z = torch.randn_like(x)
        x_mean = x + drift * dt
        x = x_mean + diffusion * torch.sqrt(dt) * z

    return x.clamp(-1, 1)