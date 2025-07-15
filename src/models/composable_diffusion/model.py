import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.slot_diffusion.SlotDiffusionUNet import SlotDiffusionUNet
from src.models.slot_diffusion.Attention import Attention

class SlotModule(nn.Module):
    """Simple slot-specific UNet."""
    def __init__(self, channels=3, dim=64):
        super().__init__()
        self.unet = SlotDiffusionUNet(dim=dim, channels=channels)

    def forward(self, x):
        return self.unet(x)

class ComposableDiffusionModel(nn.Module):
    """Factorized diffusion model with three composable slots."""
    def __init__(self, base_dim=64, channels=3):
        super().__init__()
        self.shape_slot = SlotModule(channels, base_dim)
        self.color_slot = SlotModule(channels, base_dim)
        self.box_slot = SlotModule(channels, base_dim)
        self.merge_attn = Attention(base_dim)
        self.gating_linear = nn.Linear(channels, base_dim)

    def forward(self, x, t=None):
        shape = self.shape_slot(x)
        color = self.color_slot(x)
        box = self.box_slot(x)
        merged = self.merge([shape, color, box])
        return {"shape": shape, "color": color, "box": box, "merged": merged}

    def merge(self, slots):
        stacked = torch.stack(slots, dim=1)  # B x S x C x H x W
        pooled = stacked.mean(dim=(-1, -2))  # B x S x C
        features = self.gating_linear(pooled)
        attn_out = self.merge_attn(features)
        weights = torch.softmax(attn_out.mean(dim=-1), dim=1)  # B x S
        weights = weights[:, :, None, None, None]
        return (stacked * weights).sum(dim=1)

    def sample_slots(self, shape, color, box):
        return self.merge([shape, color, box])