import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from src.models.composable_diffusion import ComposableDiffusionModel
from src.models.vanilla.ema import EMA
from src.utils.calculations import q_sample


def train(cfg, dirs, model: ComposableDiffusionModel, ema: EMA, loader, logger, device):
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer.params)
    scaler = GradScaler()
    for epoch in range(1, cfg.training.epochs + 1):
        for batch in loader:
            x = batch["image"].to(device)
            noise = torch.randn_like(x)
            t = torch.randint(0, cfg.diffusion.timesteps, (x.size(0),), device=device).long()
            x_noisy = q_sample(x, t, noise, cfg.diffusion.timesteps, cfg.diffusion.beta_start, cfg.diffusion.beta_end)
            optimizer.zero_grad()
            with autocast():
                out = model(x_noisy, t)
                loss_shape = F.mse_loss(out["shape"], noise)
                loss_color = F.mse_loss(out["color"], noise)
                loss_box = F.mse_loss(out["box"], noise)
                loss = loss_shape + loss_color + loss_box
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()
        ema.apply_shadow()
        ema.restore()