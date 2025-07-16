import torch.nn as nn

class Normalize(nn.Module):
    """
    A normalization layer that wraps torch.nn.GroupNorm.
    This is used to standardize the inputs to layers within the network,
    which helps stabilize training and improve model performance.
    """
    def __init__(self, in_channels, num_groups=32):
        """
        Initializes the Normalize layer.

        Args:
            in_channels (int): The number of input channels.
            num_groups (int): The number of groups to separate the channels into for Group Normalization.
                              The default of 32 is a common choice.
        """
        super().__init__()
        # GroupNorm is chosen as it is independent of batch size, making it effective
        # for a wide range of batch sizes, unlike BatchNorm.
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The normalized tensor with the same shape as the input.
        """
        return self.norm(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
