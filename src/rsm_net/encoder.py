"""
Shared convolutional encoder for multi-domain continual learning.

Produces fixed-size feature vectors from images of varying sizes
and channel counts (grayscale or RGB). Trained on task 0, frozen after.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvEncoder(nn.Module):
    """
    2-layer conv encoder that adapts to 1 or 3 input channels.

    Architecture:
        Conv2d(in_ch, 16, 3, pad=1) -> ReLU -> MaxPool2d(2)
        Conv2d(16, 32, 3, pad=1) -> ReLU -> MaxPool2d(2)
        AdaptiveAvgPool2d(4) -> Flatten -> Linear(32*4*4, out_features)

    AdaptiveAvgPool ensures fixed output regardless of input spatial size.
    """

    def __init__(self, out_features: int = 512, in_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels

        # Channel adapter: project any channel count to expected
        self.channel_adapt = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(32 * 4 * 4, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, C, H, W) with C = 1 or 3
        Returns:
            features: (batch, out_features)
        """
        # Adapt channels if needed
        if x.size(1) != self.in_channels:
            if x.size(1) == 1 and self.in_channels == 3:
                x = x.expand(-1, 3, -1, -1)
            elif x.size(1) == 3 and self.in_channels == 1:
                x = x.mean(dim=1, keepdim=True)

        h = self.pool(F.relu(self.channel_adapt(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = self.adaptive_pool(h)
        h = h.view(h.size(0), -1)
        return self.fc(h)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
