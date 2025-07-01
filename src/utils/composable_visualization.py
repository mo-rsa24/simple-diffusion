from typing import Dict
import matplotlib.pyplot as plt
import torch


def visualize_composition(slots: Dict[str, torch.Tensor], combined: torch.Tensor):
    fig, axes = plt.subplots(1, len(slots) + 1, figsize=(3 * (len(slots) + 1), 3))
    for i, (name, tensor) in enumerate(slots.items()):
        img = tensor.detach().cpu().squeeze().permute(1, 2, 0)
        axes[i].imshow(img.clamp(0, 1))
        axes[i].set_title(name)
        axes[i].axis('off')
    img = combined.detach().cpu().squeeze().permute(1, 2, 0)
    axes[-1].imshow(img.clamp(0, 1))
    axes[-1].set_title('combined')
    axes[-1].axis('off')
    plt.tight_layout()
    plt.show()