import torch
from matplotlib import pyplot as plt
from torchvision.transforms.v2.functional import to_pil_image

from src.dataset.utils import tensor_to_pil
from src.registry.mappings import BBOX_COLOR_MAP


def get_digit_bbox(img_tensor: torch.Tensor, threshold=0.02, padding: int = 2):
    """
    Given a tensor image of shape (C, H, W), returns the bounding box (x_min, y_min, x_max, y_max)
    enclosing non-background pixels.
    """
    C, H, W = img_tensor.shape
    # Convert to grayscale if needed
    if img_tensor.size(0) == 3:
        grayscale = img_tensor.mean(dim=0)  # shape: (H, W)
    else:
        grayscale = img_tensor.squeeze(0)  # shape: (H, W)

    # Threshold to binary mask (foreground = 1)
    mask = (grayscale > threshold)

    # Find nonzero regions (digit area)
    coords = mask.nonzero(as_tuple=False)  # shape: (N, 2)
    if coords.numel() == 0:
        print("No foreground detected!")
        print("img_tensor shape:", img_tensor.shape)
        print("img_tensor stats: min", img_tensor.min().item(), "max", img_tensor.max().item())
        print("grayscale stats: min", grayscale.min().item(), "max", grayscale.max().item())
        print("Unique grayscale values:", torch.unique(grayscale))
        return (0, 0, 0, 0)  # fallback if empty

    y_min, x_min = coords.min(dim=0).values.tolist()
    y_max, x_max = coords.max(dim=0).values.tolist()
    # Apply padding and clip to bounds
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W - 1, x_max + padding)
    y_max = min(H - 1, y_max + padding)
    return (x_min, y_min, x_max, y_max)


def draw_bbox(img, bbox, label=None, width=1, mean=0.1307, std=0.3081):
    from PIL import ImageDraw
    # img = tensor_to_pil(image_tensor, mean, std)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    color = BBOX_COLOR_MAP[label] if label is not None else "red"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def show_grid_with_bboxes(dataset, n=4):
    samples = [dataset[i] for i in range(n)]
    images = []
    titles = []

    for sample in samples:
        img = draw_bbox(sample["image"], sample["bbox"], label=sample["color_label"])
        images.append(img)
        digit = sample["digit_label"]
        color = sample["color_label"]
        titles.append(f"D:{digit} C:{color}")

    rows = int(n ** 0.5)
    cols = (n + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for ax, img, title in zip(axes.flatten(), images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

