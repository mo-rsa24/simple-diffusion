import torch
from typing import Tuple


def compute_occlusion_mask(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Return a binary mask highlighting overlapping regions of two boxes."""
    xa1, ya1, xa2, ya2 = box_a.unbind(-1)
    xb1, yb1, xb2, yb2 = box_b.unbind(-1)
    x1 = torch.max(xa1, xb1)
    y1 = torch.max(ya1, yb1)
    x2 = torch.min(xa2, xb2)
    y2 = torch.min(ya2, yb2)
    mask = (x2 > x1) & (y2 > y1)
    return mask.float()

def box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two boxes."""
    xa1, ya1, xa2, ya2 = box_a.unbind(-1)
    xb1, yb1, xb2, yb2 = box_b.unbind(-1)
    inter_w = (torch.min(xa2, xb2) - torch.max(xa1, xb1)).clamp(min=0)
    inter_h = (torch.min(ya2, yb2) - torch.max(ya1, yb1)).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / (union + 1e-8)

def color_consistency(img: torch.Tensor, digit_mask: torch.Tensor,
                      target_rgb: Tuple[float, float, float]) -> float:
    """Check mean RGB error inside the digit mask."""
    digit_pixels = img.permute(1, 2, 0)[digit_mask]
    target = torch.tensor(target_rgb, device=img.device)
    return torch.mean(torch.abs(digit_pixels.float() - target.float())).item()