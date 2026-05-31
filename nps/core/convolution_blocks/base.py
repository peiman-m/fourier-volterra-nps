from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseConvolutionBlock(nn.Module, ABC):
    """Abstract base class for convolution blocks."""

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


class SequentialBlock(nn.Sequential):
    """A Sequential container."""

    def forward(self, input, **kwargs):
        if len(self) > 0:
            # Pass kwargs only to the first module
            input = self[0](input, **kwargs)

            # Process remaining modules without kwargs
            for module in self[1:]:
                input = module(input)

        return input


class ResidualBlock(nn.Module):
    """Block of a residual network.

    Args:
        layer1 (object): Layer in the first branch.
        layer2 (object): Layer in the second branch.
        layer_post (object): Layer after adding the output of the two branches.

    Attributes:
        layer1 (object): Layer in the first branch.
        layer2 (object): Layer in the second branch.
        layer_post (object): Layer after adding the output of the two branches.
    """

    def __init__(
        self,
        layer1: Any,
        layer2: Any,
        layer_post: Any,
    ):
        super().__init__()

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer_post = layer_post

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.layer_post(
            self.layer1(*args, **kwargs) + self.layer2(*args, **kwargs)
        )
