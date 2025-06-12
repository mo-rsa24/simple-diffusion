import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset.FashionMNIST import FashionMNISTDataset
from src.models.diffusion import p_losses, sample
from src.models.unet import Unet
from src.utils.checkpoint import num_to_groups

device = "cuda" if torch.cuda.is_available() else "cpu"


# hyperparameters
image_size = 28
channels = 1
timesteps = 300
batch_size = 128

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000


num_workers  = 4
pin_memory   = True  # set False if not using a GPU

# instantiate datasets
train_ds = FashionMNISTDataset(split="train")
test_ds  = FashionMNISTDataset(split="test")

# create dataloaders
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

from torchvision.utils import save_image

epochs = 6

for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
      optimizer.zero_grad()

      batch_size = batch.shape[0]
      batch = batch.to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()

      loss = p_losses(model, batch, t, loss_type="huber")

      if step % 100 == 0:
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()
      break

    # Perform batch visualization here: Save to directory, (optional) save to wandb and/or tensorboard
    # 1. Create a side-by-side figure of real vs generated for a single image
    # 2. (Optional) For wandb, create a table of generated images
    # 3. (Optional) For tensorboard, create a grid of generated images
    break
# sample 64 images
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)

# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
