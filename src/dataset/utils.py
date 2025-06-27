from src.dataset.ChestXRay import ChestXrayDataset
from src.dataset.transforms import build_preprocessing, safe_augmentation
from typing import Dict, Tuple
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms,datasets
import torch

def load_dataset(root_dir, task: str, normalization: str='minmax', resize_strategy: str='center_crop', hist_eq: bool=False, aug_risk: str='low'):
    transform = build_preprocessing(normalization, resize_strategy, hist_eq) if aug_risk == "none" else None
    aug = safe_augmentation(aug_risk, normalization=normalization)
    return ChestXrayDataset(root_dir, task, transform=transform, aug=aug)

def unnormalize(tensor, mean=0.1307, std=0.3081):
    # Works for both grayscale and RGB
    if tensor.dim() == 3:
        mean = torch.as_tensor(mean, device=tensor.device)[..., None, None]
        std = torch.as_tensor(std, device=tensor.device)[..., None, None]
        return tensor * std + mean
    else:
        return tensor * std + mean

def tensor_to_pil(img_tensor, mean=0.1307, std=0.3081):
    # Unnormalize if needed
    img_tensor = unnormalize(img_tensor, mean, std)
    img_tensor = img_tensor.clamp(0, 1)
    img_np = (img_tensor * 255).byte().cpu().numpy()
    # [C,H,W] → [H,W,C]
    if img_np.shape[0] == 1:
        img_np = img_np[0]
    else:
        img_np = img_np.transpose(1,2,0)
    from PIL import Image
    return Image.fromarray(img_np)


def get_mnist_loaders(name: str = "MNIST", batch_size: int = 128):
    """Return train/val/test DataLoaders for MNIST or FashionMNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),                 # (0,1) → tensor
        transforms.Normalize((0.1307,), (0.3081,))  # μ, σ of MNIST
    ])
    Dataset = getattr(datasets, name)
    train_ds = Dataset(root=".data", train=True, download=True, transform=transform)
    test_ds  = Dataset(root=".data", train=False, download=True, transform=transform)

    train_len = int(0.9 * len(train_ds))
    val_len   = len(train_ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_len, val_len])

    loader = lambda ds, shuffle: DataLoader(ds, batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)


def get_colored_loaders(dataset:str = "MNIST", variant: str="foreground", batch_size: int = 128, root="./data", number: int = None, task: str = "classify")-> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoaders for MNIST or FashionMNIST."""
    if task == "classify":
        transform = transforms.Compose([
            transforms.ToTensor(),                 # (0,1) → tensor
            transforms.Normalize((0.1307,), (0.3081,))  # μ, σ of MNIST
        ])
    elif task == "generate":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2. - 1.)
        ])
    if dataset == "MNIST_BBOX":
        from src.dataset.ColoredMNISTWithBBox import ColoredMNISTWithBBox
        train_ds = ColoredMNISTWithBBox(root=root, train=True, variant=variant, number=number, transform=transform)
        test_ds = ColoredMNISTWithBBox(root=root, train=False, variant=variant, number=number, transform=transform)
    else:
        from src.dataset.ColoredMNIST import ColoredMNIST
        train_ds = ColoredMNIST(root=root, train=True, variant=variant, number=number, transform=transform)
        test_ds = ColoredMNIST(root=root, train=False, variant=variant, number=number, transform=transform)

    train_len = int(0.9 * len(train_ds))
    val_len   = len(train_ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_len, val_len])

    loader = lambda ds, shuffle: DataLoader(ds, batch_size, shuffle=shuffle, num_workers=4, pin_memory=False)
    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

def get_separate_loader(dataset: str = "MNIST", batch_size=8, number: int = None, task: str = "classify")-> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
    dataset = {
        "fg": get_colored_loaders(dataset=dataset, variant="foreground", number=number, batch_size=batch_size, task=task),
        "bg": get_colored_loaders(dataset=dataset, variant="background", number=number, batch_size=batch_size, task=task),
    }
    return dataset

# 🔵 For Foreground-Colored MNIST:
def color_foreground(image, color_rgb):
    image = np.array(image)
    mask = image >= 0.5  # digit = 1
    color_img = np.zeros((*image.shape, 3), dtype=np.uint8)
    for c in range(3):
        color_img[:, :, c] = np.where(mask, color_rgb[c], 0)  # background black
    return color_img

# 🟥 For Background-Colored MNIST:
def color_background(image, color_rgb):
    """Image: [28,28], returns [28,28,3]"""
    image = np.array(image)
    mask = image < 0.5  # background = 0
    color_img = np.zeros((*image.shape, 3), dtype=np.uint8)
    for c in range(3):
        color_img[:, :, c] = np.where(mask, color_rgb[c], 255)  # digit remains white
    return color_img


from torchvision import transforms


def get_transform(
        task: str,
        config: dict,
        use_bbox: bool = False,
        image_size: int = 28,
        normalize_mode: str = "mnist",  # or "minmax" or "diffusion"
):
    # Unpack config values
    resize_strategy = config.get('resize_strategy', 'center_crop')
    hist_eq = config.get('hist_eq', False)
    augmentation = config.get('augmentation', 'none')
    normalization = config.get('normalization', 'mnist')

    ops = []

    # 1. Resize/Crop
    if resize_strategy == "resize":
        ops.append(transforms.Resize(image_size))
    elif resize_strategy == "center_crop":
        ops.append(transforms.CenterCrop(image_size))

    # 2. Augmentation (only for classification)
    if task == "classify" and augmentation != "none":
        if augmentation in ("low", "medium"):
            ops.append(transforms.RandomCrop(image_size, padding=2))
            ops.append(transforms.RandomRotation(degrees=10))
        if augmentation == "high":
            ops.append(transforms.RandomHorizontalFlip())
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            # add more as needed

    # 3. Histogram equalization
    if hist_eq:
        from torchvision.transforms import functional as F
        ops.append(transforms.Lambda(lambda img: F.equalize(img)))

    # 4. ToTensor
    ops.append(transforms.ToTensor())

    # 5. Normalization
    if normalization == "mnist":
        ops.append(transforms.Normalize((0.1307,), (0.3081,)))
    elif normalization == "minmax":
        # Do nothing (already in [0,1])
        pass
    elif normalization == "diffusion":
        # [-1, 1] normalization
        ops.append(transforms.Lambda(lambda x: x * 2 - 1))

    # Compose and return
    return transforms.Compose(ops)
