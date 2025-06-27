# src/utils/feature_map_utils.py

import os
import torch
import matplotlib.pyplot as plt
import random

@torch.no_grad()
def visualize_feature_maps(model, input_batch, device, save_dir, sample_id=0):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    features = {}
    hooks = []

    # Example: only visualizing the bottleneck layer if it exists
    layers_to_hook = {"bottleneck": getattr(model, "downs", [None])[-1]} if hasattr(model, "downs") else {}

    def get_hook(name):
        def hook_fn(module, input, output):
            features[name] = output.detach().cpu()
        return hook_fn

    for name, layer in layers_to_hook.items():
        if layer is not None:
            hooks.append(layer.register_forward_hook(get_hook(name)))

    idx = random.randint(0, input_batch.size(0) - 1)
    _ = model(input_batch.to(device))

    for name, feat in features.items():
        fmap = feat[idx]
        fig, axes = plt.subplots(1, min(4, fmap.size(0)), figsize=(12, 4))
        for i in range(min(4, fmap.size(0))):
            axes[i].imshow(fmap[i], cmap='viridis')
            axes[i].set_title(f"{name} | Ch {i}")
            axes[i].axis('off')
        fig.tight_layout()
        path = os.path.join(save_dir, f"featuremap_{name}_epoch_{sample_id}.png")
        plt.savefig(path)
        plt.close(fig)

    for h in hooks:
        h.remove()
