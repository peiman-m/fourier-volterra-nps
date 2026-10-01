import copy
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..interpolate import Interpolator
from .base import BaseConvolution


class ConvNd(BaseConvolution):
    """
    N-dimensional convolution layer with additional features:
    - Weight activation function
    - Output size control
    - Proper initialization
    - Support for 1D, 2D, and 3D convolutions

    Output Size:
    - By default (with odd kernel sizes and padding=kernel//2), the spatial dimensions
      of the output will be the same as the input: (N, C_out, *spatial_dims)
    - With stride > 1, output spatial dimensions will be (spatial_dims // stride)
    - If transposed=True, output spatial dimensions will be (spatial_dims * stride)
    - If out_size is specified in forward(), the output will be resized to match it

    Args:
            spatial_dim: Dimensionality of the convolution (1, 2, or 3)
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel: Kernel size (must be odd for same-padding)
            stride: Stride for the convolution (default: 1)
            padding: Padding size (defaults to kernel // 2 for same-padding)
            dilation: Dilation factor for the convolution (default: 1)
            groups: Number of groups for grouped convolution (default: 1)
            bias: Whether to include a bias term
            transposed: Whether to use transposed convolution
            output_padding: Additional padding for transposed convolution
            weights_activation: Optional activation function to apply to convolution weights
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int | None = None,
        padding: int | None = None,
        dilation: int | None = None,
        groups: int | None = None,
        bias: bool = True,
        transposed: bool = False,
        output_padding: int | None = None,
        weights_activation: (
            Callable[[torch.Tensor], torch.Tensor] | nn.Module | None
        ) = None,
    ):
        if spatial_dim not in {1, 2, 3}:
            raise ValueError(
                "Only spatial_dim = [1, 2, 3] is supported, "
                f"but got spatial_dim={spatial_dim}"
            )
        if kernel % 2 != 1:
            raise ValueError("Kernel size must be odd to achieve same-padding.")

        super().__init__(
            spatial_dim=spatial_dim, in_channels=in_channels, out_channels=out_channels
        )

        self.kernel = kernel
        self.stride = stride or 1
        self.padding = padding or kernel // 2  # Use same-padding by default
        self.dilation = dilation or 1
        self.groups = groups or 1
        self.bias = bias
        self.transposed = transposed

        # Only set `output_padding` if it is given.
        additional_args = {}
        if output_padding is not None:
            additional_args["output_padding"] = output_padding

        self.conv = getattr(
            nn, f"Conv{'Transpose' if transposed else ''}{spatial_dim}d"
        )(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            **additional_args,
        )

        # Weight activation function
        self.weights_activation = (
            (lambda w: w)
            if weights_activation is None
            else copy.deepcopy(weights_activation)
        )

        # Initialize interpolation parameters
        self.interpolator = Interpolator()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, *, out_size: int | tuple[int, ...] | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the convolution layer.

        Args:
            x: Input tensor [batch_size, in_channels, *spatial_dims]
            out_size: Optional target output size

        Returns:
            Convolved tensor [batch_size, out_channels, *out_spatial_dims]

        Output Size Details:
            - With default settings (padding=kernel//2), the output spatial dimensions
              will match the input: [batch_size, out_channels, *spatial_dims]
            - With stride > 1, output dimensions will be reduced:
                [batch_size, out_channels, *spatial_dims//stride]
            - For transposed conv with stride > 1, output dimensions will be increased:
                [batch_size, out_channels, *spatial_dims*stride]
            - If out_size is provided, the output will be resized to:
                [batch_size, out_channels, *out_size]
        """
        # Apply convolution
        if not self.transposed:
            # Apply activation to weights for standard convolution
            weight = self.weights_activation(self.conv.weight)
            bias = self.conv.bias

            # Use functional for convolving with transformed weights
            out = getattr(F, f"conv{self.spatial_dim}d")(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            out = self.conv(x)

        # Resize output if requested
        if out_size is not None:
            out = self.interpolator(out, out_size)

        return out
