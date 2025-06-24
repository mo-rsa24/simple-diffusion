import torch

from src.utils.visualization import visualize_images

# Example with MNIST dummy batch
dummy = torch.randn(128, 1, 28, 28)
visualize_images(dummy, num_images=4)

# Example with a single RGB image
rgb_img = torch.randn(3, 256, 256)
visualize_images(rgb_img)

# Example with normalization reversal
visualize_images(dummy, denormalize=True, mean=[0.1307], std=[0.3081])

# Save to disk
visualize_images(dummy, save_path="outputs/sample_grid.png")
