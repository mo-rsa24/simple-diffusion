# models/autoencoder.py
import torch
import torch.nn as nn

class SimpleConvEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),           # 14 -> 7
            nn.ReLU(),
            nn.ConvTranspose2d(64, latent_dim, 2, 2), # 7 -> 14
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1),  # keep spatial
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 0),  # 14 -> 12
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, 5, 1, 0),  # 12 -> 8
        )

        # ←—— ADD THESE THREE LINES ——→
        self.out_dim = latent_dim
        self.spatial_h = 8
        self.spatial_w = 8
    def forward(self, x):
        return self.encoder(x)



class SimpleConvDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, 2, 1),   # 8x8 → 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),           # 16x16 → 32x32
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 5, 1, 0),          # 32x32 → 28x28 (crop to match MNIST)
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)

