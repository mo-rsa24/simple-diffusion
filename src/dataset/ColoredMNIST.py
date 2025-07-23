from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
from src.dataset.utils import color_foreground, color_background
from src.registry.mappings import COLOR_MAP


class ColoredMNIST(Dataset):
    def __init__(self, root, train=True, download=True, variant="foreground", color_map=COLOR_MAP, transform=None, number: int = None
                 , digit_color_label:int=None, bbox_color_label:int=None):
        self.mnist = MNIST(root=root, train=train, download=download)
        self.variant = variant
        self.color_map = color_map
        self.transform = transform or transforms.ToTensor()
        self.fixed_digit_color_label = digit_color_label
        self.fixed_bbox_color_label = bbox_color_label
        if number is not None:
            mask = self.mnist.targets == number
            self.mnist.data = self.mnist.data[mask]
            self.mnist.targets = self.mnist.targets[mask]

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, digit_label = self.mnist[idx]
        color_label = self.fixed_digit_color_label if self.fixed_digit_color_label is not None else random.randint(0, 9)
        rgb = self.color_map[color_label]

        if self.variant == "foreground":
            img_colored = color_foreground(img, rgb)
        else:
            img_colored = color_background(img, rgb)

        img_pil = Image.fromarray(img_colored)
        img_tensor = self.transform(img_pil)  # shape: [3, 28, 28]

        return {
            "image": img_tensor,
            "digit_label": digit_label,
            "color_label": digit_label  # identity mapping; can decouple for compositionality
        }