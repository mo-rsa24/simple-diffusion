import random

from pandas.tests.tools.test_to_numeric import transform
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from src.dataset.utils import color_foreground, color_background
from src.detectors import get_digit_bbox, draw_bbox
from src.registry.mappings import COLOR_MAP


class ColoredMNISTWithBBox(Dataset):
    def __init__(self, root, train=True, download=True, variant="foreground", color_map=COLOR_MAP, transform=None, number:int = None):
        self.mnist = MNIST(root=root, train=train, download=download)
        self.variant = variant
        self.color_map = color_map
        self.transform = transform or transforms.ToTensor()

        if number is not None:
            mask = self.mnist.targets == number
            self.mnist.data = self.mnist.data[mask]
            self.mnist.targets = self.mnist.targets[mask]

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, digit_label = self.mnist[idx]
        color_label = digit_label
        digit_rgb = self.color_map[color_label]

        # Apply colorization
        if self.variant == "foreground":
            img_colored = color_foreground(img, digit_rgb)
        else:
            img_colored = color_background(img, digit_rgb)

        # Transform to tensor
        img_pil = Image.fromarray(img_colored)
        img_tensor = transforms.ToTensor()(img_pil)

        # Compute bbox from the colored tensor
        bbox = get_digit_bbox(img_tensor)

        # Choose bbox_label different from color_label
        possible_labels = list(set(self.color_map.keys()) - {color_label})
        bbox_label = random.choice(possible_labels)

        # Draw bbox on the image using bbox_label's color
        img_with_bbox = draw_bbox(img_pil, bbox, label=bbox_label)

        if self.transform:
            final_tensor = self.transform(img_with_bbox)
        else:
            final_tensor = transforms.ToTensor()(img_with_bbox)
        return {
            "image": final_tensor,
            "idx": idx,
            "digit_label": digit_label,
            "color_label": color_label,
            "bbox_label": bbox_label,
            "bbox": bbox,
        }