import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
# Add these helper classes to SlotDiffusionUNet.py

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SinusoidalPositionEmbeddings(nn.Module):
    """ Converts integer timesteps into a continuous embedding vector. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Upsample(nn.Module):
    """ An upsampling layer with a convolution. """
    def __init__(self, dim, out_dim=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, default(out_dim, dim), 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    """ A downsampling layer with a convolution. """
    def __init__(self, dim, out_dim=None):
        super().__init__()
        self.rearrange = Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        self.conv = nn.Conv2d(dim * 4, default(out_dim, dim), 1)

    def forward(self, x):
        return self.conv(self.rearrange(x))

# In ResidualBlock.py

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None): # Add time_emb_dim
        super().__init__()
        # --- Time MLP ---
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if exists(time_emb_dim)
            else None
        )
        # --- Convolutions ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    # --- Update forward to accept time 't' ---
    def forward(self, x, t=None):
        h = self.norm1(self.conv1(x))

        # Add time embedding
        if exists(self.time_mlp) and exists(t):
            time_emb = self.time_mlp(t)
            h = h + rearrange(time_emb, "b c -> b c 1 1")

        h = F.silu(h)
        h = self.norm2(self.conv2(h))
        return h + self.skip(x)


# In SlotDiffusionUNet.py - this is the final, updated class
from einops.layers.torch import Rearrange


class SlotDiffusionUNet(nn.Module):
    def __init__(self, dim=64, channels=1, dim_mults=(1, 2, 4, 8)):
        super().__init__()

        # --- Initial Convolution ---
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)

        # --- Time Embedding ---
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- UNet Architecture ---
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # --- Downsampling Path ---
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    ResidualBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        # --- Middle Path ---
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        # self.mid_attn = Attention(mid_dim) # Attention is optional but recommended
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # --- Upsampling Path ---
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    ResidualBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    ResidualBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        # --- Final Convolution ---
        self.final_conv = nn.Sequential(
            ResidualBlock(dim * 2, dim, time_emb_dim=time_dim),
            nn.Conv2d(dim, channels, 1)
        )

    def forward(self, x, time, self_cond=None):  # Add 'time' to forward
        x = self.init_conv(x)
        r = x.clone()  # Residual connection

        # Create time embedding
        t = self.time_mlp(time)

        h = []

        # Downsampling
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        # Middle
        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = upsample(x)

        # Final layers
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)