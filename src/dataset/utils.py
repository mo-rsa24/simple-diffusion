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


def get_colored_loaders(dataset:str = "MNIST", variant: str="foreground", batch_size: int = 128, root="./data")-> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoaders for MNIST or FashionMNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),                 # (0,1) → tensor
        transforms.Normalize((0.1307,), (0.3081,))  # μ, σ of MNIST
    ])
    if dataset == "MNIST_BBOX":
        from src.dataset.ColoredMNISTWithBBox import ColoredMNISTWithBBox
        train_ds = ColoredMNISTWithBBox(root=root, train=True, variant=variant, transform=transform)
        test_ds = ColoredMNISTWithBBox(root=root, train=False, variant=variant, transform=transform)
    else:
        from src.dataset.ColoredMNIST import ColoredMNIST
        train_ds = ColoredMNIST(root=root, train=True, variant=variant, transform=transform)
        test_ds = ColoredMNIST(root=root, train=False, variant=variant, transform=transform)

    train_len = int(0.9 * len(train_ds))
    val_len   = len(train_ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_len, val_len])

    loader = lambda ds, shuffle: DataLoader(ds, batch_size, shuffle=shuffle, num_workers=4, pin_memory=False)
    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

def get_separate_loader(dataset: str = "MNIST", batch_size=8)-> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
    dataset = {
        "fg": get_colored_loaders(dataset=dataset, variant="foreground", batch_size=batch_size),
        "bg": get_colored_loaders(dataset=dataset, variant="background", batch_size=batch_size),
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
