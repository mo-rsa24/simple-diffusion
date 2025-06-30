import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math


# ----------------------
# Attention Block
# ----------------------
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        dots = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = dots.softmax(dim=-1)
        out = attn @ v
        return self.out(out)