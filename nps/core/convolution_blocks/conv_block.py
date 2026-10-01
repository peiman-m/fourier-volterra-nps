import copy
from typing import Any

import torch
import torch.nn as nn

from ...utils.helpers import compress_batch_dimensions
from ..convolutions import ConvNd
from .base import BaseConvolutionBlock, ResidualBlock


class ConvBlock(BaseConvolutionBlock):
    """A flexible standard convolutional block.
    Reference: https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses/coders/nn.py

    This block supports standard convolutions, transposed convolutions,
    depthwise separable convolutions, and residual connections.

    Args:
        spatial_dim: Dimensionality of the convolution (1D, 2D, 3D).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel: Kernel size(s).
        stride: Stride for the convolution. Defaults to 1.
        padding: Padding for the convolution. Defaults to kernel // 2.
        groups: Number of groups for grouped convolution. Defaults to 1.
        transposed: Whether to use transposed convolution. Defaults to False.
        activation: Activation function to use. Defaults to ReLU.
        separable: Whether to use depthwise separable convolution. Defaults to False.
        residual: Whether to use residual connections. Defaults to True.
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel: int | tuple[int, ...],
        stride: int | None = None,
        padding: int | tuple[int, ...] | None = None,
        groups: int | None = None,
        transposed: bool = False,
        activation: nn.Module | None = None,
        separable: bool = False,
        residual: bool = True,
    ):
        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        if residual:
            self.net = self._init_residual(
                spatial_dim=spatial_dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                transposed=transposed,
                activation=activation,
                separable=separable,
            )
        else:
            if separable:
                self.net = self._init_separable_conv(
                    spatial_dim=spatial_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    transposed=transposed,
                    activation=activation,
                )
            else:
                self.net = self._init_conv(
                    spatial_dim=spatial_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    transposed=transposed,
                    activation=activation,
                )

    def _calculate_output_padding(self, stride) -> dict[str, Any]:
        """Calculate output padding for transposed convolutions."""
        if not stride or stride == 1:
            return {}

        if isinstance(stride, int):
            if stride % 2 == 0:
                return {"output_padding": stride // 2}
            raise ValueError(
                f"Stride={stride} must be even for transposed convolutions."
            )

        # Handle tuple case
        if any(s > 1 and s % 2 != 0 for s in stride):
            odd_dims = [i for i, s in enumerate(stride) if s > 1 and s % 2 != 0]
            raise ValueError(
                f"Stride values > 1 must be even. Problematic dimensions: {odd_dims}"
            )

        padding_tuple = tuple(s // 2 if s > 1 else 0 for s in stride)
        return {"output_padding": padding_tuple}

    def _init_conv(
        self,
        *,
        spatial_dim,
        in_channels,
        out_channels,
        kernel,
        stride,
        padding,
        groups,
        transposed,
        activation,
    ) -> nn.Sequential:
        """Initialize a standard convolution block."""
        # Create network layers list
        net = []

        # Add activation if provided
        if activation:
            net.append(copy.deepcopy(activation))

        # Calculate output padding for transposed convolutions
        output_padding = (
            self._calculate_output_padding(stride) if transposed and stride != 1 else {}
        )

        # Add convolution layer
        net.append(
            ConvNd(
                spatial_dim=spatial_dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                transposed=transposed,
                **output_padding,
            )
        )

        return nn.Sequential(*net)

    def _init_separable_conv(
        self,
        *,
        spatial_dim,
        in_channels,
        out_channels,
        kernel,
        stride,
        padding,
        transposed,
        activation,
    ) -> nn.Sequential:
        """Initialize a depthwise separable convolution.

        Creates two convolution layers:
        1. A depthwise convolution (groups=in_channels)
        2. A pointwise convolution (kernel=1) to change the number of channels
        """
        return nn.Sequential(
            # Depthwise convolution (spatial filtering)
            self._init_conv(
                spatial_dim=spatial_dim,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
                groups=in_channels,
                transposed=transposed,
                activation=activation,
            ),
            # Pointwise convolution (channel mixing)
            self._init_conv(
                spatial_dim=spatial_dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=1,
                stride=1,
                padding=0,
                groups=1,
                transposed=False,
                activation=None,
            ),
        )

    def _init_residual(
        self,
        *,
        spatial_dim,
        in_channels,
        out_channels,
        kernel,
        stride,
        padding,
        groups,
        transposed,
        activation,
        separable,
    ) -> ResidualBlock:
        """Initialize a residual block.

        Creates a residual connection with optional input transformation
        if dimensions don't match.
        """
        # Use smaller channel count for the intermediate layers
        intermediate_channels = min(in_channels, out_channels)

        # Determine if we need input transformation
        needs_transform = in_channels != intermediate_channels

        # Also need transform if stride is not 1
        if stride:
            if isinstance(stride, int):
                needs_transform = needs_transform or (stride != 1)
            else:
                needs_transform = needs_transform or any(s != 1 for s in stride)

        # Create input transform function
        if needs_transform:
            # Use 1x1 convolution for input transformation
            input_transform = self._init_conv(
                spatial_dim=spatial_dim,
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel=1,
                stride=stride,
                padding=0,
                groups=1,
                transposed=transposed,
                activation=None,
            )
        else:
            input_transform = lambda x: x

        # Create the residual block
        return ResidualBlock(
            # Input transformation path
            input_transform,
            # Main processing path
            nn.Sequential(
                # First convolution
                ConvBlock(
                    spatial_dim=spatial_dim,
                    in_channels=in_channels,
                    out_channels=intermediate_channels,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    transposed=transposed,
                    activation=activation,
                    separable=separable,
                    residual=False,
                ),
                # Pointwise (1x1) convolution for channel mixing
                self._init_conv(
                    spatial_dim=spatial_dim,
                    in_channels=intermediate_channels,
                    out_channels=intermediate_channels,
                    kernel=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    transposed=False,
                    activation=activation,
                ),
            ),
            # Output projection
            self._init_conv(
                spatial_dim=spatial_dim,
                in_channels=intermediate_channels,
                out_channels=out_channels,
                kernel=1,
                stride=1,
                padding=0,
                groups=1,
                transposed=False,
                activation=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, uncompress = compress_batch_dimensions(x, self.spatial_dim + 1)
        return uncompress(self.net(x))
