# src/models/vanilla/composable_unet.py

import torch
import torch.nn as nn


class ComposableExpertUnet(nn.Module):
    """
    A UNet that composes two expert UNets.
    The expert UNets are frozen during composition.
    """

    def __init__(self, unet_a, unet_b, fusion_type='additive'):
        """
        Initializes the ComposableUNet.
        Args:
            unet_a (nn.Module): The first expert UNet.
            unet_b (nn.Module): The second expert UNet.
            fusion_type (str): The method to fuse the outputs ('additive' or 'gating').
        """
        super().__init__()

        # Freeze the expert models
        self.unet_a = unet_a
        for param in self.unet_a.parameters():
            param.requires_grad = False

        self.unet_b = unet_b
        for param in self.unet_b.parameters():
            param.requires_grad = False

        self.fusion_type = fusion_type

        if self.fusion_type == 'gating':
            # A simple gating network. Assumes the output of the UNets has a certain number of channels.
            # This might need adjustment based on the actual UNet architecture.
            # Let's assume the output channel size is accessible via unet_a.channels
            in_channels = unet_a.channels  # Or whatever the output channel dim is
            self.gating_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 2, kernel_size=1),
                nn.Softmax(dim=1)
            )
        elif self.fusion_type != 'additive':
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def forward(self, x, t):
        """
        Forward pass for the ComposableUNet.
        Args:
            x (torch.Tensor): The input tensor (noisy image).
            t (torch.Tensor): The timestep tensor.
        Returns:
            torch.Tensor: The predicted noise.
        """
        # Get the noise predictions from both experts
        noise_a = self.unet_a(x, t)
        noise_b = self.unet_b(x, t)

        if self.fusion_type == 'additive':
            # Simple additive composition
            return noise_a + noise_b

        elif self.fusion_type == 'gating':
            # The gating network needs some input to decide the weights.
            # A simple approach is to use the noisy image `x` itself.
            # A more complex approach could use intermediate features.
            gate_input = torch.cat((noise_a, noise_b), dim=1)
            weights = self.gating_net(gate_input)

            # Apply the learned weights to combine the noise predictions
            return weights[:, 0:1, :, :] * noise_a + weights[:, 1:2, :, :] * noise_b
