from pathlib import Path
from datetime import timedelta
from typing   import Dict
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim    import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# project utilities ───────────────────────────────────────────
from src.config.configs import Config
from src.models.vanilla.ema import EMA
from src.models.vanilla.unet import Unet
from src.train.logging.training_logger_utils import (
    log_training_start, log_epoch_start, log_batch,
    log_epoch_summary, visualize_epoch,
    log_json, log_training_end
)
from src.utils.checkpoint_manager import CheckpointManager
from src.monitoring.email_alert_mailtrap import alert_on_success


# ──────────────────────────────────────────────────────────────
#  EDM helpers (Karras et al. 2022)
# ──────────────────────────────────────────────────────────────
class EDMNoiseSchedule:
    """
    Log-uniform σ sampling as in Karras et al.:
        σ = (σ_max^{ρ} + u (σ_min^{ρ}-σ_max^{ρ}))^{1/ρ}
        with u ~ U(0,1)  and ρ≈7 for MNIST-scale tasks.
    """
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min, self.sigma_max, self.rho = sigma_min, sigma_max, rho

    def sample(self, batch, device):
        u = torch.rand(batch, device=device)
        return (self.sigma_max ** self.rho +
                u * (self.sigma_min ** self.rho - self.sigma_max ** self.rho)
               ) ** (1.0 / self.rho)


def edm_loss(model, x0, sigma, sigma_data=0.5):
    """
    Karras et al. loss (Eq. 12):  w · ‖ fθ(x,σ) − x₀ ‖²,
    where  x = x₀ + σ ε,     ε~𝓝(0,I),
           w = (σ² + σ_data²) / (σ σ_data)²
    """
    noise  = torch.randn_like(x0)
    x      = x0 + sigma[:, None, None, None] * noise
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

    with autocast():
        pred = model(x, sigma)        # network must take (x, σ)
        loss = 0.5 * weight[:, None, None, None] * (pred - x0).pow(2)

    return loss.mean()


# ──────────────────────────────────────────────────────────────
#  Heun (2-nd order) ODE sampler with optional stochasticity
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def edm_sampler(model,
                shape,
                sigma_min=0.002,
                sigma_max=80.0,
                rho=7,
                S_churn=0,
                S_tmin=0,
                S_tmax=float('inf'),
                S_noise=1,
                steps=40):
    """
    Deterministic Heun integrator by default.
    Set S_churn>0 for stochasticity (Section C.2).
    """
    device = next(model.parameters()).device
    batch, C, H, W = shape
    sigmas = torch.linspace(0, 1, steps+1, device=device)
    sigmas = (sigma_max ** (1/rho) +
              sigmas * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    x = torch.randn(shape, device=device) * sigmas[0]

    for i in range(steps):
        sigma   = sigmas[i]
        sigma_n = sigmas[i+1]
        # stochastic churn
        if S_churn > 0 and sigma_n >= S_tmin and sigma_n <= S_tmax:
            gamma = min(S_churn / steps, 2 ** 0.5 - 1)
            eps   = torch.randn_like(x) * S_noise
            x     = x + eps * torch.sqrt((sigma_n ** 2 - sigma_n ** 2) * gamma)

        # Heun step
        d       = model(x, sigma)      # predicts x₀
        d       = (x - d) / sigma      # score ∝ ε
        x_hat   = x + (sigma_n - sigma) * d            # Euler
        d_prime = (x_hat - model(x_hat, sigma_n)) / sigma_n
        x       = x + 0.5 * (sigma_n - sigma) * (d + d_prime)

    return x.clamp(-1, 1)

# src/models/edm/edm_unet.py
class EDMUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = Unet(**kwargs)        # reuse your file
    def forward(self, x, sigma):
        # Karras preconditioning
        c_in  = 1 / torch.sqrt(sigma**2 + 1)
        c_skip= sigma / torch.sqrt(sigma**2 + 1)
        x_in  = c_in[:, None, None, None] * x
        h     = self.net(x_in, sigma)    # assumes net embeds σ internally
        return c_skip[:, None, None, None] * x + h
