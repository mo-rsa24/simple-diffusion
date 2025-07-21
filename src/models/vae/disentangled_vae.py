import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A simple residual block for the decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))


class DisentangledVAE(nn.Module):
    """
    A Beta-VAE with a split latent space and an optional ResNet-based decoder
    to encourage robust spatial and statistical disentanglement.
    """

    def __init__(self, in_channels=3, latent_dim_digit=32, latent_dim_bbox=16, use_resnet=True):
        super().__init__()
        self.latent_dim_digit = latent_dim_digit
        self.latent_dim_bbox = latent_dim_bbox
        self.total_latent_dim = latent_dim_digit + latent_dim_bbox

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu_digit = nn.Linear(128 * 4 * 4, latent_dim_digit)
        self.fc_logvar_digit = nn.Linear(128 * 4 * 4, latent_dim_digit)

        self.fc_mu_bbox = nn.Linear(128 * 4 * 4, latent_dim_bbox)
        self.fc_logvar_bbox = nn.Linear(128 * 4 * 4, latent_dim_bbox)

        # --- Decoder ---
        self.decoder_input = nn.Linear(self.total_latent_dim, 128 * 4 * 4)

        if use_resnet:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
                ResidualBlock(128, 64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
                ResidualBlock(64, 32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
                nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:  # The original, simpler decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )

    def encode(self, x):
        result = self.encoder(x)
        return (self.fc_mu_digit(result), self.fc_logvar_digit(result)), \
            (self.fc_mu_bbox(result), self.fc_logvar_bbox(result))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 128, 4, 4)
        return self.decoder(result)

    def forward(self, x):
        (mu_digit, logvar_digit), (mu_bbox, logvar_bbox) = self.encode(x)
        z_digit = self.reparameterize(mu_digit, logvar_digit)
        z_bbox = self.reparameterize(mu_bbox, logvar_bbox)
        z_combined = torch.cat([z_digit, z_bbox], dim=1)
        reconstruction = self.decode(z_combined)

        return {
            "reconstruction": reconstruction,
            "mu_digit": mu_digit, "logvar_digit": logvar_digit, "z_digit": z_digit,
            "mu_bbox": mu_bbox, "logvar_bbox": logvar_bbox, "z_bbox": z_bbox,
        }
