import torch
from torch import nn
from functools import partial
from einops import rearrange, reduce
import torch.nn.functional as F
from src.models.vanilla.attention import LinearAttention, Attention
from src.models.vanilla.embeddings import SinusoidalPositionEmbeddings
from src.models.vanilla.helpers import exists, default, Residual, Downsample, Upsample
from src.models.vanilla.normalization import PreNorm


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight standardized convolution.
    https://arxiv.org/abs/1903.10520
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(
            x, normalized_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )


class Block(nn.Module):
    """Standard ResNet block with GroupNorm."""

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """ResNet block with time and conditional embeddings."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, cond_emb_dim=None, groups=8):
        super().__init__()

        # Combined dimension for time and conditional embeddings
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim + cond_emb_dim, dim_out * 2))
            if exists(time_emb_dim) and exists(cond_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, combined_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(combined_emb):
            combined_emb = self.mlp(combined_emb)
            combined_emb = rearrange(combined_emb, "b c -> b c 1 1")
            scale_shift = combined_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class CompositionalVisualGeneration(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,
            use_attention_at=(),
            num_experts=3,
            num_classes=11,  # 0-9 digits + 1 null token
            num_colors=11  # 0-9 colors + 1 null token
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.use_attention_at = set(use_attention_at)
        self.num_experts = num_experts

        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # --- Conditional and Time Embeddings ---
        time_dim = dim * 4
        cond_dim = dim  # Dimension for each conditional embedding

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Embedding layers for digit class, digit color, and bbox color
        self.digit_class_emb = nn.Embedding(num_classes, cond_dim)
        self.digit_color_emb = nn.Embedding(num_colors, cond_dim)
        self.bbox_color_emb = nn.Embedding(num_colors, cond_dim)

        # We'll concatenate these, so the total conditional dimension is 3 * cond_dim
        total_cond_dim = cond_dim * 3

        # This block will fuse time and conditional embeddings
        block_klass = partial(ResnetBlock, time_emb_dim=time_dim, cond_emb_dim=total_cond_dim,
                              groups=resnet_block_groups)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            attn_block = Attention(dim_in) if ind in self.use_attention_at else LinearAttention(dim_in)
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in),
                    Residual(PreNorm(dim_in, attn_block)),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            attn_block = Attention(dim_out) if ind in self.use_attention_at else LinearAttention(dim_out)
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out),
                    Residual(PreNorm(dim_out, attn_block)),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        self.out_dim = default(out_dim, channels)

        # The final block now needs to handle the combined embeddings
        final_block_klass = partial(ResnetBlock, time_emb_dim=time_dim, cond_emb_dim=total_cond_dim,
                                    groups=resnet_block_groups)
        self.final_res_block = final_block_klass(dim * 2, dim)

        # The final convolution outputs predictions for all experts
        self.final_conv = nn.Conv2d(dim, self.out_dim * self.num_experts, 1)

    def forward(self, x, time, digit_labels, digit_color_labels, bbox_color_labels, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        # --- Create and Combine Embeddings ---
        t = self.time_mlp(time)
        d_emb = self.digit_class_emb(digit_labels)
        dc_emb = self.digit_color_emb(digit_color_labels)
        bc_emb = self.bbox_color_emb(bbox_color_labels)

        # Concatenate all conditional embeddings
        cond_emb = torch.cat([d_emb, dc_emb, bc_emb], dim=1)

        # Concatenate time and conditional embeddings for the ResNet blocks
        combined_emb = torch.cat([t, cond_emb], dim=1)

        h = []
        for block, attn, downsample in self.downs:
            x = block(x, combined_emb)
            h.append(x)
            x = attn(x)
            x = downsample(x)

        x = self.mid_block1(x, combined_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, combined_emb)

        for block, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, combined_emb)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, combined_emb)

        # Returns a tensor of shape (B, C * num_experts, H, W)
        return self.final_conv(x)
