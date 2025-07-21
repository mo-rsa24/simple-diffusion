import random
import torch
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

from src.dataset.utils import color_foreground, color_background
from src.detectors import get_digit_bbox, draw_bbox
from src.registry.mappings import COLOR_MAP


class ComposableColoredMNISTWithBBox(Dataset):
    """
    MNIST with independently sampled digit color and bounding box color.
    Now includes binary masks for the digit and the bounding box to enable
    training of disentangled expert models.
    """

    def __init__(self, root, train=True, download=True, variant="foreground", color_map=COLOR_MAP, transform=None,
                 number: int = None, digit_color_label:int=None, bbox_color_label:int=None):
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
        """
        Gets an item from the dataset.

        Returns:
            dict: A dictionary containing:
                - "image": The final image tensor with colored digit and bbox.
                - "digit_label": The label of the MNIST digit.
                - "color_label": The label for the digit's color.
                - "bbox_label": The label for the bounding box's color.
                - "bbox": The coordinates of the bounding box.
                - "digit_mask": A binary mask tensor for the digit pixels.
                - "bbox_mask": A binary mask tensor for the bounding box pixels.
        """
        img, digit_label = self.mnist[idx]
        color_label = self.fixed_digit_color_label if self.fixed_digit_color_label is not None else random.randint(0, 9)
        bbox_label = self.fixed_bbox_color_label if self.fixed_bbox_color_label is not None else random.randint(0, 9)
        digit_rgb = self.color_map[color_label]

        # --- Mask Generation (from PIL image) ---
        # Create a binary mask of the digit from the raw MNIST image
        digit_mask_pil = img.point(lambda p: 255 if p > 20 else 0, 'L').convert('1')

        # Color the image based on the variant
        if self.variant == "foreground":
            img_colored = color_foreground(img, digit_rgb)
        else:
            img_colored = color_background(img, digit_rgb)

        img_pil = Image.fromarray(img_colored)

        # Get bounding box coordinates
        img_tensor_for_bbox = transforms.ToTensor()(img_pil)
        bbox = get_digit_bbox(img_tensor_for_bbox)

        # --- BBox Mask Generation ---
        bbox_mask_pil = Image.new('1', img_pil.size, 0)
        bbox_draw = ImageDraw.Draw(bbox_mask_pil)
        bbox_draw.rectangle(bbox, outline=1, width=1)

        # Draw the colored bounding box on the main image
        img_with_bbox = draw_bbox(img_pil, bbox, label=bbox_label)

        # --- Finalize Masks and Tensors ---
        # Ensure masks do not overlap (digit is foreground)
        bbox_mask_np = np.array(bbox_mask_pil)
        digit_mask_np = np.array(digit_mask_pil)
        bbox_mask_np[digit_mask_np] = 0  # Subtract digit area from bbox mask

        # Convert final image to tensor
        if self.transform:
            final_tensor = self.transform(img_with_bbox)
        else:
            final_tensor = transforms.ToTensor()(img_with_bbox)

        # Convert masks to tensors
        digit_mask_tensor = transforms.ToTensor()(Image.fromarray(digit_mask_np))
        bbox_mask_tensor = transforms.ToTensor()(Image.fromarray(bbox_mask_np))

        return {
            "image": final_tensor,
            "digit_label": digit_label,
            "color_label": color_label,
            "bbox_label": bbox_label,
            "bbox": bbox,
            "digit_mask": digit_mask_tensor,
            "bbox_mask": bbox_mask_tensor,
        }
