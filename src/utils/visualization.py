import os
import random
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

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


import matplotlib.pyplot as plt
from pathlib import Path
import torch

def save_side_by_side_images(real: torch.Tensor,
                             generated: torch.Tensor,
                             side_by_side_dir: Path,
                             input_range: str = "auto"):
    """
    Save paired real vs. generated images side by side.
    Supports both grayscale (1×H×W) and RGB (3×H×W) CHW tensors.
    """
    n = min(len(real), len(generated))
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    if input_range == "auto":
        is_diffusion = generated.min() < 0
    elif input_range == "diffusion":
        is_diffusion = True
    else:
        is_diffusion = False

    for i in range(n):
        # 1) Pull out the i-th sample
        real_img = real[i].detach().cpu()
        gen_img  = generated[i].detach().cpu()

        def normalize_img(t: torch.Tensor) -> torch.Tensor:
            if t.min() < 0 or is_diffusion:
                t = t.clamp(-1, 1)
                t = (t + 1) / 2
            else:
                t = t.clamp(0, 1)
            return t

        real_img = normalize_img(real_img)
        gen_img  = normalize_img(gen_img)

        # 3) Convert CHW → HWC or HW depending on channels
        def chw_to_display(img: torch.Tensor):
            C, H, W = img.shape
            if C == 1:
                # grayscale: squeeze to (H, W)
                arr = img.squeeze(0).numpy()
                cmap = "gray"
            elif C == 3:
                # RGB: permute to (H, W, 3)
                arr = img.permute(1, 2, 0).numpy()
                cmap = None
            else:
                raise ValueError(f"Unsupported channel count: {C}")
            return arr, cmap

        real_arr, real_cmap = chw_to_display(real_img)
        gen_arr,  gen_cmap  = chw_to_display(gen_img)

        # 4) Plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].imshow(real_arr, cmap=real_cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_title("Real");      axes[0].axis("off")

        axes[1].imshow(gen_arr, cmap=gen_cmap,  vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title("Generated"); axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(side_by_side_dir / Path(f"sample_{i:03d}_comparison.png"))
        plt.close(fig)

def visualize_predictions(model,
                          dirs: dict,
                          loader,
                          device,
                          epoch: int = None,
                          n: int = 9,
                          mean: tuple = None,
                          std: tuple = None):
    import torch
    model.eval()
    samples_dir = Path(dirs.get("results_samples")) / f"epoch_{epoch}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(loader))
    images = batch["image"][:n].to(device)
    label_keys = [k for k in batch if k.endswith("_label")]
    # Move labels to cpu and slice
    batch_labels = {k: batch[k][:n] for k in label_keys}

    with torch.no_grad():
        logits_dict = model(images)
        # Turn logits into predictions
        preds_dict = {k: v.argmax(1).cpu() for k, v in logits_dict.items()}

    images = images.cpu()
    rows = int(n ** 0.5)
    cols = (n + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i in range(n):
        img = images[i]
        if mean is not None and std is not None:
            m = torch.tensor(mean).view(-1, 1, 1)
            s = torch.tensor(std).view(-1, 1, 1)
            img = img * s + m
        elif img.min() < 0:
            img = (img + 1) / 2
        img = img.clamp(0, 1)

        if img.shape[0] == 1:  # grayscale
            img = img.squeeze(0)
        else:
            img = img.permute(1, 2, 0)
        axes[i].imshow(img)
        # Flexible: Build label string dynamically
        title = []
        for logit_key, pred in preds_dict.items():
            label_type = logit_key.replace("_logits", "")
            gt_key = f"{label_type}_label"
            if gt_key in batch_labels:
                title.append(f"{label_type[:1].upper()}_pred: {pred[i]} (GT:{batch_labels[gt_key][i].item()})")
        axes[i].set_title("\n".join(title), fontsize=8)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(samples_dir / "comparison_grid.png")
    plt.close(fig)




def visualize_with_bbox(sample, bbox_dir):
    img = sample["image"].permute(1, 2, 0).numpy()  # (H, W, C)
    bbox = sample["bbox"].numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw rectangle
    x_min, y_min, x_max, y_max = bbox
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title(f"Digit: {sample['digit_label']}, Color: {sample['color_label']}")
    fig.tight_layout()
    fig.savefig(bbox_dir / Path(f"digit_label_{sample['digit_label']}_color_{sample['color_label']}.png"))
    plt.close(fig)
    plt.show()


def visualize_feature_maps(
    model,
    sample,
    selected_layers=None,
    save_dir=None,
    n_display=16  # Number of channels to display per layer
):
    """
    Visualizes feature maps from selected Conv2d layers in a model.
    """
    model.eval()
    fmap, hooks = {}, []

    def make_hook(name):
        def hook(layer, input, output):
            fmap[name] = output.detach().cpu()
        return hook

    # Register forward hooks
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            if selected_layers is None or name in selected_layers:
                hooks.append(layer.register_forward_hook(make_hook(name)))

    # Ensure sample is batched
    sample = sample.unsqueeze(0) if sample.dim() == 3 else sample
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = sample.to(device)
    with torch.no_grad():
        _ = model(sample)

    for h in hooks:
        h.remove()

    # Visualize each collected feature map
    for name, feat in fmap.items():
        channels = feat.shape[1]
        display_channels = min(n_display, channels)
        ncols = 8
        nrows = math.ceil(display_channels / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
        axs = axs.flatten()

        vmin = feat.min().item()
        vmax = feat.max().item()

        for i in range(display_channels):
            axs[i].imshow(feat[0, i], cmap='magma', vmin=vmin, vmax=vmax)
            axs[i].axis('off')

        for j in range(display_channels, len(axs)):
            axs[j].axis('off')

        fig.suptitle(f"Feature maps — {name}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/featuremap_{name}.png", bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def tsne_plot(model, loader, n=1000, zoom=0.5, perplexity=30, seed=42, save_path=None):
    """
    Generates a t-SNE plot of learned CNN representations with superimposed images.

    Args:
        model: Trained CNN model.
        loader: Dataloader for dataset.
        n: Number of samples to visualize.
        zoom: Scale factor for thumbnail images.
        perplexity: t-SNE perplexity.
        seed: Random seed for reproducibility.
        save_path: Path to save the plot (optional).
    """
    model.eval()
    feats, labels, images = [], [], []

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(next(model.parameters()).device)
            h = model.features(x).view(x.size(0), -1)
            feats.append(h.cpu())
            labels.append(y)
            images.extend(x.cpu())  # store the raw images
            if len(images) >= n:
                break

    X = TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=seed).fit_transform(torch.cat(feats)[:n])
    Y = torch.cat(labels)[:n]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("t-SNE of Learned Representations")

    # Normalize for better layout
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X_norm = (X - x_min) / (x_max - x_min)

    # Add thumbnails
    for i in range(n):
        image = images[i].squeeze(0).numpy()  # single channel MNIST
        imagebox = OffsetImage(image, cmap='gray', zoom=zoom)
        ab = AnnotationBbox(imagebox, X_norm[i], frameon=False)
        ax.add_artist(ab)

    scatter = ax.scatter(X_norm[:, 0], X_norm[:, 1], c=Y, cmap="tab10", alpha=0.2, s=1)
    fig.colorbar(scatter, ax=ax)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()



def visualize_images(
    data,
    num_images: int = 4,
    denormalize: bool = False,
    mean: list = [0.5],
    std: list = [0.5],
    save_path: str = None,
    title: str = None,
    show: bool = True,
):
    # --- Convert input to tensor ---
    if isinstance(data, Image.Image):
        data = torch.from_numpy(np.array(data)).permute(2, 0, 1).float() / 255
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    data = data.clone().detach()  # Avoid modifying original

    # --- Normalize shape ---
    if data.ndim == 2:
        data = data.unsqueeze(0)  # (H, W) → (1, H, W)

    if data.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        if data.shape[0] <= 4:  # (C, H, W)
            batch = data.unsqueeze(0)  # (1, C, H, W)
        else:  # (H, W, C)
            data = data.permute(2, 0, 1)
            batch = data.unsqueeze(0)
    elif data.ndim == 4:
        batch = data
    else:
        raise ValueError(f"Unsupported input shape: {data.shape}")

    # --- Slice if needed ---
    batch = batch[:num_images]

    # --- De-normalize if required ---
    if denormalize:
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        batch = batch * std + mean

    # --- Create grid ---
    grid_img = make_grid(batch, nrow=min(num_images, int(np.sqrt(num_images))), normalize=False)

    # --- Convert for plotting ---
    img = grid_img.permute(1, 2, 0).cpu().numpy()

    # Grayscale
    if img.shape[2] == 1:
        img = img.squeeze(-1)
        cmap = "gray"
    else:
        cmap = None

    # --- Plot ---
    plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(img, cmap=cmap)

    # --- Save or Show ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def apply_gradcam(model, sample, class_idx=None, target_layer="features.3"):
    model.eval()
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
    sample = sample.unsqueeze(0).to(next(model.parameters()).device)

    out = model(sample)
    pred = out.argmax(dim=1).item()
    class_idx = class_idx if class_idx is not None else pred

    # Extract CAM
    activation_map = cam_extractor(class_idx, out)

    # Overlay on image
    fig, ax = plt.subplots()
    img = to_pil_image(sample.squeeze(0).cpu())
    ax.imshow(img, cmap='gray')
    ax.imshow(activation_map[0].permute(1,2,0).cpu().numpy(), cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(f"Grad-CAM for class {class_idx}")
    plt.show()

def visualize_misclassifications(model, test_loader, max_vis=4, per_class_limit=2, target_layer="features.3"):
    from collections import defaultdict

    model.eval()
    device = next(model.parameters()).device
    class_counter = defaultdict(int)
    total_count = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        out = model(data)
        pred = out.argmax(dim=1)

        incorrect = pred != target
        incorrect_indices = torch.where(incorrect)[0]

        for idx in incorrect_indices:
            true_class = int(target[idx])
            pred_class = int(pred[idx])

            if class_counter[true_class] >= per_class_limit:
                continue

            print(f"[❌] Misclassified: True={true_class}, Pred={pred_class}")
            apply_gradcam(model, data[idx], class_idx=pred_class, target_layer=target_layer)

            class_counter[true_class] += 1
            total_count += 1

            if total_count >= max_vis:
                print(f"\n✅ Reached visualization cap: {max_vis}")
                return
