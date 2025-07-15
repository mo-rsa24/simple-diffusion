import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision

from src.dataset.ComposableColoredMNISTWithBBox import ComposableColoredMNISTWithBBox
from src.models.composable_diffusion import ComposableDiffusionModel
from src.utils.calculations import q_sample
from src.debug.composable_debug.filters import tiny_subset
from torchvision.utils import save_image


class DebugPipeline:
    """Step-by-step debugging utilities for ComposableDiffusionModel."""

    def __init__(self, root: str, out_dir: str = "debug_outputs", batch_size: int = 4):
        self.root = root
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _get_loader(self, num_examples: int = 8) -> DataLoader:
        dataset = ComposableColoredMNISTWithBBox(
            root=self.root,
            train=True,
            download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]),
        )
        subset = tiny_subset(dataset, num_examples)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    def sanity_check_overfit(self, epochs: int = 5) -> torch.nn.Module:
        loader = self._get_loader()
        model = ComposableDiffusionModel(base_dim=32, channels=3).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for epoch in range(epochs):
            for batch in loader:
                img = batch["image"].to(self.device)
                noise = torch.randn_like(img)
                t = torch.randint(0, 50, (img.size(0),), device=self.device).long()
                x_noisy = q_sample(img, t, noise, timesteps=50, beta_start=1e-4, beta_end=0.02)
                # out = model(x_noisy, t)["merged"]
                # loss = torch.nn.functional.mse_loss(out, noise)
                import torch.nn.functional as F
                out = model(x_noisy, t)
                loss_shape = F.mse_loss(out["shape"], noise)
                loss_color = F.mse_loss(out["color"], noise)
                loss_box = F.mse_loss(out["box"], noise)
                loss = loss_shape + loss_color + loss_box
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
                print(f"[Epoch {epoch}] loss={loss.item():.4f}")
        loss_plot = torch.tensor(losses)
        save_image(loss_plot.unsqueeze(0), str(self.out_dir / "loss_plot.png"))
        return model

    # ------------------------------------------------------------------
    def single_timestep(self, model: ComposableDiffusionModel) -> None:
        loader = self._get_loader(num_examples=1)
        batch = next(iter(loader))
        img = batch["image"].to(self.device)
        noise = torch.randn_like(img)
        t = torch.tensor([10], device=self.device)
        x_noisy = q_sample(img, t, noise, timesteps=50, beta_start=1e-4, beta_end=0.02)
        pred = model(x_noisy, t)["merged"]
        x_prev = x_noisy - pred  # very rough approx
        out_dir = self.out_dir / "step4"
        out_dir.mkdir(exist_ok=True)
        save_image((img + 1) / 2, str(out_dir / "x_start.png"))
        save_image((x_noisy + 1) / 2, str(out_dir / "x_noisy.png"))
        save_image((x_prev + 1) / 2, str(out_dir / "x_prev.png"))


__all__ = ["DebugPipeline"]