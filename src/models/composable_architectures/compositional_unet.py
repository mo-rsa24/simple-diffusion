import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# --- Helper Functions and Modules (These are correct) ---

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SinusoidalPositionEmbeddings(nn.Module):
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
    def __init__(self, dim, out_dim=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(dim, default(out_dim, dim), 3, padding=1)
    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        self.conv = nn.Conv2d(dim, default(out_dim, dim), 4, 2, 1)
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None):
        super().__init__()
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if exists(time_emb_dim)
            else None
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, t=None, film_params=None):
        h = self.norm1(self.conv1(x))
        if exists(self.time_mlp) and exists(t):
            time_emb = self.time_mlp(t)
            h = h + rearrange(time_emb, "b c -> b c 1 1")
        if exists(film_params):
            scale, shift = film_params
            h = h * scale.unsqueeze(-1).unsqueeze(-1) + shift.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.norm2(self.conv2(h))
        return h + self.skip(x)

# --- Core Model Components ---

class SharedUNet(nn.Module):
    # --- THIS IS THE FINAL, ROBUST, AND CORRECTED VERSION ---
    def __init__(self, dim, channels, dim_mults=(1, 2, 4)):
        super().__init__()
        self.channels = channels
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Downsampling Path
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResidualBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Downsample(dim_out, dim_out) if not is_last else nn.Identity()
            ]))

        # Middle Path
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling Path
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResidualBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ResidualBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Upsample(dim_in, dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Conv2d(dim, channels, 1)

        gating_dims = []
        for d1, d2, _ in self.downs:
            gating_dims.extend([d1.conv1.out_channels, d2.conv1.out_channels])
        gating_dims.extend([self.mid_block1.conv1.out_channels, self.mid_block2.conv1.out_channels])
        for u1, u2, _ in self.ups:
            gating_dims.extend([u1.conv1.out_channels, u2.conv1.out_channels])
        self.layer_dims_for_gating = gating_dims

    def forward(self, x, time, film_params):
        x = self.init_conv(x)
        t = self.time_mlp(time)
        h = []
        film_idx = 0

        for block1, block2, downsample in self.downs:
            x = block1(x, t, film_params[film_idx]); film_idx += 1
            x = block2(x, t, film_params[film_idx]); film_idx += 1
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, film_params[film_idx]); film_idx += 1
        x = self.mid_block2(x, t, film_params[film_idx]); film_idx += 1

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, film_params[film_idx]); film_idx += 1
            x = block2(x, t, film_params[film_idx]); film_idx += 1
            x = upsample(x)

        return self.final_conv(x)

class GatingNetwork(nn.Module):
    # (No changes needed)
    def __init__(self, concept_dim, layer_dims):
        super().__init__()
        self.networks = nn.ModuleList()
        for dim in layer_dims:
            self.networks.append(nn.Sequential(
                nn.Linear(concept_dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim * 2)
            ))
    def forward(self, concepts):
        film_params = []
        for net in self.networks:
            params = net(concepts)
            scale, shift = params.chunk(2, dim=-1)
            film_params.append((scale, shift))
        return film_params

class CompositionalUNet(nn.Module):
    # (No changes needed)
    def __init__(self, concept_dim=3, unet_dim=64, unet_channels=3, unet_dim_mults=(1, 2, 4)):
        super().__init__()
        self.shared_unet = SharedUNet(dim=unet_dim, channels=unet_channels, dim_mults=unet_dim_mults)
        gating_layer_dims = self.shared_unet.layer_dims_for_gating
        self.gating_network = GatingNetwork(concept_dim, gating_layer_dims)

    def forward(self, x, time, concepts):
        predictions = []
        for concept in concepts:
            concept_batch = concept.repeat(x.size(0), 1)
            film_params = self.gating_network(concept_batch)
            prediction = self.shared_unet(x, time, film_params)
            predictions.append(prediction)
        merged_prediction = torch.stack(predictions, dim=0).mean(dim=0)
        return {
            "shape": predictions[0],
            "color": predictions[1],
            "box": predictions[2],
            "merged": merged_prediction
        }