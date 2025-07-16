import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from src.config.configs import Config
from src.dataset.transforms import mnist_transform


class MNISTDataset(Dataset):
    def __init__(self, root_dir="./data", split="train", number:int = None, transform=None):
        assert split in {"train", "val", "test"}, "Split must be one of train/val/test"
        # Use test set for both val and test for simplicity
        is_train = split == "train"
        self.dataset = datasets.MNIST(
            root=root_dir,
            train=is_train,
            download=True,
            transform=transform)

        # Optionally, use a validation split
        if number is not None:
            mask = self.dataset.targets == number
            self.dataset.data = self.dataset.data[mask]
            self.dataset.targets = self.dataset.targets[mask]
        if split == "val":
            self.dataset.data = self.dataset.data[:5000]
            self.dataset.targets = self.dataset.targets[:5000]
        if split == "test":
            self.dataset.data = self.dataset.data[5000:]
            self.dataset.targets = self.dataset.targets[5000:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, targets = self.dataset[idx]  # Ignore label, use only image
        return {
            "image": img,
            "digit_label": targets,
        }


def get_mnist_loaders(cfg: Config, number: int = None):
    transform = mnist_transform(cfg)
    dataset = MNISTDataset(root_dir=cfg.dataset.data_dir, split="train", number=number, transform=transform)
    if cfg.sanity_checks.debug:
        from src.dataset.utils import tiny_subset
        dataset = tiny_subset(dataset, cfg.sanity_checks.num_examples)
    val_dataset = MNISTDataset(root_dir=cfg.dataset.data_dir, split="val", number=number, transform=transform)
    if cfg.sanity_checks.debug:
        from src.dataset.utils import tiny_subset
        val_dataset = tiny_subset(val_dataset, cfg.sanity_checks.num_examples)
    test_dataset = MNISTDataset(root_dir=cfg.dataset.data_dir, split="test", number=number, transform=transform)

    train_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers)

    return train_loader, val_loader, test_loader