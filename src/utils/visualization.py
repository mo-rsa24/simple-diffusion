from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# use seed for reproducability
torch.manual_seed(0)

def plot(image, imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


def show_image(img, title=None):
    """
    Display an image from various formats using matplotlib.pyplot.imshow().

    Args:
        img: A PyTorch tensor, NumPy array, PIL image, or torchvision grid output.
        title (str, optional): Optional title for the plot.
    """
    # Convert PIL to numpy
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Convert torch tensor to numpy
    elif isinstance(img, torch.Tensor):
        if img.ndim == 4:
            # Assume batched tensor: take first image
            img = img[0]

        if img.ndim == 3:
            # CHW → HWC if channels-first
            if img.shape[0] in [1, 3]:  # Likely CHW
                img = img.permute(1, 2, 0)
            elif img.shape[-1] in [1, 3]:  # Possibly already HWC
                pass
            else:
                raise ValueError(f"Ambiguous 3D tensor shape: {img.shape}")

        elif img.ndim == 2:
            # Grayscale, already 2D
            pass

        elif isinstance(img, torch.Tensor):
            raise ValueError(f"Unsupported tensor shape: {img.shape}")

        # Convert to numpy
        img = img.detach().cpu().numpy()

    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # Likely CHW
            img = np.transpose(img, (1, 2, 0))

    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # Squeeze singleton channels (e.g., grayscale [H, W, 1] → [H, W])
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=2)

    # Normalize float images if needed (matplotlib expects 0–1 floats or 0–255 ints)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 1)

    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def save_side_by_side_images(real, generated, side_by_side_dir):
    n = min(len(real), len(generated))
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].imshow(real[i].cpu().squeeze(), cmap="gray")
        axes[0].set_title("Real"); axes[0].axis("off")
        axes[1].imshow(generated[i].cpu().squeeze(), cmap="gray")
        axes[1].set_title("Generated"); axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(side_by_side_dir / Path(f"sample_{i:03d}_comparison.png"))
        plt.close(fig)