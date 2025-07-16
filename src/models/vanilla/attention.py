from einops import rearrange
from torch import nn, einsum
import torch

from src.models.vanilla.normalization import Normalize


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class AttnBlock(nn.Module):
    """
    A self-attention block that allows the model to focus on different parts of the image.
    This is a key component for capturing long-range dependencies in the data.
    """

    def __init__(self, in_channels):
        """
        Initializes the AttnBlock.

        Args:
            in_channels (int): The number of channels in the input feature map.
        """
        super().__init__()
        self.in_channels = in_channels

        # The input is normalized before the attention mechanism is applied.
        self.norm = Normalize(in_channels)

        # Linear layers to project the input into query (q), key (k), and value (v) spaces.
        # This is the core of the attention mechanism.
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # The output projection layer. This takes the attended value features and
        # projects them back to the original channel dimension.
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Applies the self-attention mechanism.

        Args:
            x (torch.Tensor): The input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: The output feature map with attention applied, with the same shape as the input.
        """
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for matrix multiplication
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # (B, H*W, C)
        k = k.reshape(b, c, h * w)  # (B, C, H*W)
        v = v.reshape(b, c, h * w)
        v = v.permute(0, 2, 1)  # (B, H*W, C)

        # Compute attention scores (w = Q * K^T)
        # The scaling factor (c**(-0.5)) is crucial for stabilizing gradients.
        w_ = torch.bmm(q, k) * (c ** -0.5)
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Apply attention scores to values (w * V)
        w_ = w_.permute(0, 2, 1)  # (B, C, H*W)
        h_ = torch.bmm(v.permute(0, 2, 1), w_)
        h_ = h_.reshape(b, c, h, w)

        # Project out and add residual connection
        h_ = self.proj_out(h_)

        return x + h_
