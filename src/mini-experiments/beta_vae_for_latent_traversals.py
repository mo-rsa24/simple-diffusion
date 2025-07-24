import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt


# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 32
    BATCH_SIZE = 128
    # We will train two models: one with 10 dims for general purpose, one with 2 for specific plots
    LATENT_DIMS_10D = 10
    LATENT_DIMS_2D = 2
    NUM_EPOCHS = 25
    LR = 1e-3
    BETA = 4.0  # Key parameter for disentanglement (β > 1)
    OUTPUT_DIR = "beta_vae_visuals"


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# --- 1. Dataset (Colored MNIST) ---
class ColoredMNIST(Dataset):
    def __init__(self, train=True):
        mnist_dataset = MNIST(root="./data", train=train, download=True, transform=ToTensor())
        self.data = mnist_dataset.data
        self.targets = mnist_dataset.targets
        self.transforms = Resize((Config.IMG_SIZE, Config.IMG_SIZE))
        # Use more distinct colors for better visualization
        self.color_map = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])  # Red, Green, Blue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].float() / 255.0
        label = self.targets[idx]
        # Assign a color based on the digit label (e.g., 0-2 red, 3-5 green, 6-9 blue)
        color_idx = label.item() // 4
        color = self.color_map[color_idx]
        img_rgb = torch.stack([img * color[0], img * color[1], img * color[2]])
        return self.transforms(img_rgb), label, color_idx


# --- 2. β-VAE Model ---
class BetaVAE(nn.Module):
    def __init__(self, latent_dims):
        super(BetaVAE, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dims)
        self.fc_log_var = nn.Linear(256, latent_dims)
        self.decoder_input = nn.Linear(latent_dims, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, x):
        result = self.encoder(x)
        return self.fc_mu(result), self.fc_log_var(result)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# --- 3. Loss & Training ---
def loss_function(recon_x, x, mu, log_var, beta):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kld_loss


def train(model, dataloader, optimizer, epoch, num_epochs):
    model.train()
    loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]", leave=False)
    for data, _, _ in loop:
        data = data.to(Config.DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var, Config.BETA)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item() / len(data))


# --- 4. Visualization Functions ---

@torch.no_grad()
def generate_reconstructions(model, dataloader, filename):
    model.eval()
    # Get a single batch from the test dataloader
    originals, _, _ = next(iter(dataloader))
    originals = originals[:16].to(Config.DEVICE)  # Take 16 samples
    recons, _, _ = model(originals)

    # Compare original and reconstructed
    comparison = torch.cat([originals, recons]).cpu()
    grid = make_grid(comparison, nrow=16, padding=4)
    save_image(grid, os.path.join(Config.OUTPUT_DIR, filename))
    print(f"✅ Saved reconstruction grid to {filename}")


@torch.no_grad()
def generate_full_traversal_grid(model, data_sample, filename):
    model.eval()
    data_sample = data_sample.to(Config.DEVICE).unsqueeze(0)
    mu, _ = model.encode(data_sample)

    all_traversals = []
    traversal_range = torch.linspace(-3, 3, 10)

    for dim_to_traverse in range(model.latent_dims):
        # Add original reconstruction as the first image in the row
        original_recon = model.decode(mu)
        all_traversals.append(original_recon)

        for val in traversal_range:
            z_traversed = mu.clone()
            z_traversed[0, dim_to_traverse] = val
            recon = model.decode(z_traversed)
            all_traversals.append(recon)

    grid = make_grid(torch.cat(all_traversals).cpu(), nrow=11, padding=4)
    save_image(grid, os.path.join(Config.OUTPUT_DIR, filename))
    print(f"✅ Saved full traversal grid to {filename}")


@torch.no_grad()
def perform_latent_arithmetic(model, dataset, filename):
    model.eval()

    # Find specific images for the analogy: "Green 5" - "Green 2" + "Red 2" = "Red 5"
    def find_instance(digit, color_idx):
        for i in range(len(dataset)):
            img, lbl, c_idx = dataset[i]
            if lbl == digit and c_idx == color_idx:
                return img.to(Config.DEVICE).unsqueeze(0)
        raise ValueError(f"Could not find instance for digit {digit} color {color_idx}")

    try:
        img_g5 = find_instance(5, 1)  # Green 5
        img_g2 = find_instance(2, 0)  # Green 2 -> Mismatch, should be red
        img_r2 = find_instance(2, 0)  # Red 2

        # Correcting the logic based on dataset color assignment
        # 0-3 Red (idx 0), 4-7 Green (idx 1), 8-9 Blue (idx 2)
        img_g5 = find_instance(5, 1)  # Green 5
        img_g2 = find_instance(4, 1)  # Green 4
        img_r4 = find_instance(1, 0)  # Red 1

        mu_g5, _ = model.encode(img_g5)
        mu_g2, _ = model.encode(img_g2)
        mu_r4, _ = model.encode(img_r4)

        # Analogy: z(Red 5) = z(Green 5) - z(Green 4) + z(Red 4)
        mu_result = mu_g5 - mu_g2 + mu_r4
        recon_result = model.decode(mu_result)

        # Get reconstructions of originals for comparison
        recon_g5, recon_g2, recon_r4 = model.decode(mu_g5), model.decode(mu_g2), model.decode(mu_r4)

        grid = make_grid(torch.cat([
            img_g5.cpu(), recon_g5.cpu(),
            img_g2.cpu(), recon_g2.cpu(),
            img_r4.cpu(), recon_r4.cpu(),
            recon_result.cpu(), recon_result.cpu()  # Show result twice for alignment
        ]), nrow=2, padding=4)
        save_image(grid, os.path.join(Config.OUTPUT_DIR, filename))
        print(f"✅ Saved latent arithmetic result to {filename}")

    except ValueError as e:
        print(f"Could not perform latent arithmetic: {e}")


@torch.no_grad()
def generate_latent_scatter(model, dataloader, filename):
    model.eval()
    if model.latent_dims != 2:
        print(f"⚠️ Skipping scatter plot: requires LATENT_DIMS=2, but model has {model.latent_dims}.")
        return

    all_mu = []
    all_labels = []
    all_colors = []
    for imgs, labels, colors in tqdm(dataloader, desc="Encoding for scatter plot", leave=False):
        mu, _ = model.encode(imgs.to(Config.DEVICE))
        all_mu.append(mu.cpu())
        all_labels.append(labels)
        all_colors.append(colors)

    all_mu = torch.cat(all_mu).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_colors = torch.cat(all_colors).numpy()

    plt.figure(figsize=(12, 10))
    # Plot by digit shape
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap='tab10', s=5, alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)), title="Digits")
    plt.title("Latent Space Colored by Digit")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f"scatter_by_digit_{filename}"))
    plt.close()

    plt.figure(figsize=(12, 10))
    # Plot by color
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_colors, cmap='viridis', s=5, alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Red', 'Green', 'Blue'], title="Colors")
    plt.title("Latent Space Colored by Color")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f"scatter_by_color_{filename}"))
    plt.close()
    print(f"✅ Saved latent space scatter plots to {filename} prefixes")


@torch.no_grad()
def generate_manifold_plot(model, filename, n_images_per_dim=20):
    model.eval()
    if model.latent_dims != 2:
        print(f"⚠️ Skipping manifold plot: requires LATENT_DIMS=2, but model has {model.latent_dims}.")
        return

    # Create a grid of latent values
    grid_x = np.linspace(-3, 3, n_images_per_dim)
    grid_y = np.linspace(-3, 3, n_images_per_dim)

    generated_images = []
    for y in grid_y:
        for x in grid_x:
            z_sample = torch.tensor([[x, y]], device=Config.DEVICE).float()
            generated_images.append(model.decode(z_sample).cpu())

    grid = make_grid(torch.cat(generated_images), nrow=n_images_per_dim, padding=4)
    save_image(grid, os.path.join(Config.OUTPUT_DIR, filename))
    print(f"✅ Saved learned manifold plot to {filename}")


# --- Main Execution ---
if __name__ == '__main__':
    # --- Part 1: Train and visualize the 10D model ---
    print("--- Training 10D model for traversals, reconstructions, and arithmetic ---")
    model_10d = BetaVAE(latent_dims=Config.LATENT_DIMS_10D).to(Config.DEVICE)
    optimizer_10d = torch.optim.Adam(model_10d.parameters(), lr=Config.LR)
    train_dataset = ColoredMNIST(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train(model_10d, train_dataloader, optimizer_10d, epoch, Config.NUM_EPOCHS)

    print("\n--- Generating visualizations for 10D model ---")
    test_dataset = ColoredMNIST(train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    # 1. Reconstructions
    generate_reconstructions(model_10d, test_dataloader, "reconstructions_10d.png")

    # 2. Full Traversal Grid
    sample_img, _, _ = test_dataset[15]  # A green '5'
    generate_full_traversal_grid(model_10d, sample_img, "full_traversal_10d.png")

    # 3. Latent Arithmetic
    perform_latent_arithmetic(model_10d, train_dataset, "latent_arithmetic_10d.png")

    # --- Part 2: Train and visualize the 2D model ---
    print("\n\n--- Training 2D model for scatter and manifold plots ---")
    model_2d = BetaVAE(latent_dims=Config.LATENT_DIMS_2D).to(Config.DEVICE)
    optimizer_2d = torch.optim.Adam(model_2d.parameters(), lr=Config.LR)

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train(model_2d, train_dataloader, optimizer_2d, epoch, Config.NUM_EPOCHS)

    print("\n--- Generating visualizations for 2D model ---")
    # 4. Latent Space Scatter Plot
    generate_latent_scatter(model_2d, test_dataloader, "plot_2d.png")

    # 5. Learned Manifold Plot
    generate_manifold_plot(model_2d, "manifold_2d.png")

    print("\n🎉 All visualizations generated in the 'beta_vae_visuals' directory!")
