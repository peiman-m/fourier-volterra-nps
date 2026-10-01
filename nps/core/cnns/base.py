from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseCNN(nn.Module, ABC):
    """Abstract base class for convolutional neural networks."""

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
