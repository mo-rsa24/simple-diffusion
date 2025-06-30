import torch.nn as nn
from src.models.slot_diffusion.ResidualBlock import ResidualBlock


# ----------------------
# Downsample / Upsample
# ----------------------
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

# ----------------------
# SlotDiffusion UNet (2-slots: digit vs background)
# ----------------------
class SlotDiffusionUNet(nn.Module):
    def __init__(self, dim=64, channels=1):
        super().__init__()
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)

        self.down1 = ResidualBlock(dim, dim)
        self.down2 = ResidualBlock(dim, dim*2)
        self.pool = Downsample(dim*2)

        self.mid = ResidualBlock(dim*2, dim*2)

        self.up1 = Upsample(dim*2)
        self.up2 = ResidualBlock(dim*2, dim)
        self.final = nn.Conv2d(dim, channels, 1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.pool(x)
        x = self.mid(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)
