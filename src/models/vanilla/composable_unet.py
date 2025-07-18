from torch import nn
import torch
from functools import partial
import torch.nn.functional as F
from einops import rearrange, reduce

from src.models.vanilla.attention import LinearAttention, Attention
from src.models.vanilla.embeddings import SinusoidalPositionEmbeddings
from src.models.vanilla.helpers import exists, default, Residual, Downsample, Upsample
from src.models.vanilla.normalization import PreNorm
from torch.utils.checkpoint import checkpoint


class ComposableUnet(nn.Module):
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
            # --- New Conditional Parameters ---
            num_classes_digit=11,  # 10 digits + 1 for null/unconditional
            num_classes_digit_color=11,
            num_classes_bbox_color=11,
            embedding_dim=256
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.use_attention_at = set(use_attention_at)

        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # --- Time and Class Embeddings ---
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # ✨ New Embedding Layers for each condition
        self.digit_emb = nn.Embedding(num_classes_digit, embedding_dim)
        self.digit_color_emb = nn.Embedding(num_classes_digit_color, embedding_dim)
        self.bbox_color_emb = nn.Embedding(num_classes_bbox_color, embedding_dim)

        # ✨ Projection layer for the combined embeddings
        # We project from embedding_dim * 3 to the time_dim to make them additive
        self.embedding_proj = nn.Linear(embedding_dim * 3, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            attn_block = Attention(dim_in) if ind in self.use_attention_at else LinearAttention(dim_in)
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, attn_block)),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            attn_block = Attention(dim_out) if ind in self.use_attention_at else LinearAttention(dim_out)
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, attn_block)),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, digit_labels, digit_color_labels, bbox_color_labels, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        # 1. Get time embedding
        t = self.time_mlp(time)

        # 2. ✨ Create and combine class embeddings
        digit_emb = self.digit_emb(digit_labels)
        digit_color_emb = self.digit_color_emb(digit_color_labels)
        bbox_color_emb = self.bbox_color_emb(bbox_color_labels)

        # Concatenate all label embeddings
        combined_labels = torch.cat([digit_emb, digit_color_emb, bbox_color_emb], dim=-1)

        # Project to the same dimension as the time embedding
        class_emb = self.embedding_proj(combined_labels)

        # 3. ✨ Add class embeddings to the time embedding
        cond = t + class_emb

        h = []

        for block, attn, downsample in self.downs:
            x = checkpoint(block, x, cond)  # Pass combined embedding
            h.append(x)
            x = checkpoint(attn, x)
            x = downsample(x)

        x = checkpoint(self.mid_block1, x, cond)  # Pass combined embedding
        x = checkpoint(self.mid_attn, x)
        x = checkpoint(self.mid_block2, x, cond)  # Pass combined embedding

        for block, attn, upsample in self.ups:
            skip = h.pop()
            x = torch.cat((x, skip), dim=1)
            x = checkpoint(block, x, cond)  # Pass combined embedding
            x = checkpoint(attn, x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = checkpoint(self.final_res_block, x, cond)  # Pass combined embedding
        return self.final_conv(x)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
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
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
