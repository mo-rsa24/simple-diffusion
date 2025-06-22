import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, split="train", image_size=28):
        assert split in {"train", "val", "test"}, "Split must be one of train/val/test"
        # Use test set for both val and test for simplicity
        is_train = split == "train"
        self.dataset = datasets.MNIST(
            root="./data",
            train=is_train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # Map [0,1] to [-1,1] as required by most diffusion models
                transforms.Lambda(lambda x: x * 2. - 1.)
            ])
        )
        # Optionally, use a validation split
        if split == "val":
            self.dataset.data = self.dataset.data[:5000]
            self.dataset.targets = self.dataset.targets[:5000]
        if split == "test":
            self.dataset.data = self.dataset.data[5000:]
            self.dataset.targets = self.dataset.targets[5000:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # Ignore label, use only image
        return img
