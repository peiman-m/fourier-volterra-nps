import copy

import torch
import torch.nn as nn

from ...utils.helpers import compress_batch_dimensions
from ..convolutions import SpectralConv
from ..interpolate import Interpolator
from ..mlp import MLP
from .base import BaseConvolutionBlock, ResidualBlock, SequentialBlock


class FourierBlock(BaseConvolutionBlock):
    """A flexible Fourier block.

    This block supports spectral convolution, depthwise separable variant,
    and residual connections.

    Args:
        spatial_dim: Dimensionality of the spectral convolution (1D, 2D, etc).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to keep along each dimension.
        groups: Number of groups for grouped convolution. Defaults to 1.
        activation: Activation function to use. Defaults to None.
        separable: Whether to use depthwise separable convolution. Defaults to False.
        residual: Whether to use residual connections. Defaults to True.
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        modes: int | tuple[int, ...],
        groups: int | None = None,
        activation: nn.Module | None = None,
        separable: bool = False,
        residual: bool = True,
    ):
        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.modes = modes

        if residual:
            self.net = self._init_residual(
                spatial_dim=spatial_dim,
                in_channels=in_channels,
                out_channels=out_channels,
                modes=modes,
                groups=groups,
                activation=activation,
                separable=separable,
            )
        else:
            if separable:
                self.net = self._init_separable_sconv(
                    spatial_dim=spatial_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    modes=modes,
                    activation=activation,
                )
            else:
                self.net = self._init_sconv(
                    spatial_dim=spatial_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    modes=modes,
                    groups=groups,
                    activation=activation,
                )

    def __repr__(self) -> str:
        return (
            f"FourierBlock(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"modes={self.modes})"
        )

    def _init_sconv(
        self,
        *,
        spatial_dim,
        in_channels,
        out_channels,
        modes,
        groups,
        activation,
    ) -> SequentialBlock:
        """Initialize a Fourier block."""

        # If out_size if given, instead of trimming or
        # padding the input, interpolation is used
        net: list[nn.Module] = [Interpolator()]

        # Add activation if provided
        if activation:
            net.append(copy.deepcopy(activation))

        net.append(
            SpectralConv(
                spatial_dim=spatial_dim,
                in_channels=in_channels,
                out_channels=out_channels,
                modes=modes,
                groups=groups,
            )
        )

        return SequentialBlock(*net)

    def _init_separable_sconv(
        self,
        *,
        spatial_dim,
        in_channels,
        out_channels,
        modes,
        activation,
    ) -> SequentialBlock:
        """Initialize a depthwise separable Fourier block.

        Creates two convolution layers:
        1. A depthwise spectral convolution (groups=in_channels)
        2. A pointwise linear mapping to change the number of channels
        """
        return SequentialBlock(
            # Depthwise convolution (spatial filtering)
            self._init_sconv(
                spatial_dim=spatial_dim,
                in_channels=in_channels,
                out_channels=in_channels,
                modes=modes,
                groups=in_channels,
                activation=activation,
            ),
            # Pointwise map (channel mixing)
            MLP(
                in_dim=in_channels,
                out_dim=out_channels,
                feature_axis=1,  # Channel dimension
            ),
        )

    def _init_residual(
        self,
        *,
        spatial_dim,
        in_channels,
        out_channels,
        modes,
        groups,
        activation,
        separable,
    ) -> ResidualBlock:
        """Initialize a residual block.

        Creates a residual connection with optional input transformation
        if dimensions don't match.
        """
        # Default activation for residual blocks if none provided
        activation = nn.GELU() if activation is None else copy.deepcopy(activation)

        # Use smaller channel count for the intermediate layers
        intermediate_channels = min(in_channels, out_channels)

        # Determine if we need input transformation
        needs_transform = in_channels != intermediate_channels

        if needs_transform:
            # Input transformation to match dimensions
            input_transform = SequentialBlock(
                Interpolator(),
                MLP(
                    in_dim=in_channels,
                    out_dim=intermediate_channels,
                    feature_axis=1,  # Channel dimension
                ),
            )
        else:
            input_transform = Interpolator()

        return ResidualBlock(
            # Input transformation path
            input_transform,
            # Main spectral convolution path
            SequentialBlock(
                # First Spectral convolution
                FourierBlock(
                    spatial_dim=spatial_dim,
                    in_channels=in_channels,
                    out_channels=intermediate_channels,
                    modes=modes,
                    groups=groups,
                    activation=activation,
                    separable=separable,
                    residual=False,
                ),
                # Linear mapping for channel
                MLP(
                    in_dim=intermediate_channels,
                    out_dim=intermediate_channels,
                    feature_axis=1,  # Channel dimension
                ),
            ),
            # Output projection
            nn.Sequential(
                MLP(
                    in_dim=intermediate_channels,
                    out_dim=intermediate_channels,
                    feature_axis=1,  # Channel dimension
                ),
                copy.deepcopy(activation),
                MLP(
                    in_dim=intermediate_channels,
                    out_dim=out_channels,
                    feature_axis=1,  # Channel dimension
                ),
            ),
        )

    def forward(
        self, x: torch.Tensor, *, out_size: int | tuple[int, ...] | None = None
    ) -> torch.Tensor:
        # Add out_size parameter if provided
        kwargs = {}
        if out_size is not None:
            kwargs["out_size"] = out_size

        x, uncompress = compress_batch_dimensions(x, self.spatial_dim + 1)
        return uncompress(self.net(x, **kwargs))
