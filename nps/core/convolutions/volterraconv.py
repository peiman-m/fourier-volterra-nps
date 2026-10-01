import copy

import einops
import torch
import torch.nn as nn

from ..interpolate import Interpolator
from .base import BaseConvolution


class VolterraConvNd(BaseConvolution):
    """
    N-dimensional convolution layer with additional features:
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
        spatial_dim (int): Dimensionality of the convolution (1, 2, or 3)
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel (int): Kernel size (must be odd for same-padding)
        stride (int): Stride for the convolution (default: 1)
        padding (int): Padding size (defaults to kernel // 2 for same-padding)
        dilation (int): Dilation factor for the convolution (default: 1)
        groups (int): Number of groups for grouped convolution (default: 1)
        bias (bool): Whether to include a bias term
        transposed: Whether to use transposed convolution
        output_padding: Additional padding for transposed convolution
        volterra_low_rank (bool): Whether to use low-rank approximation of the 2nd
            order Volterra kernel. If false, the exact 2nd order kernel is used
            (Only available for dim=1). (default: True)
        volterra_rank (int | None): Approximation rank of the 2nd order Volterra filter.
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
        volterra_low_rank: bool = False,
        volterra_rank: int | None = None,
    ):
        if spatial_dim not in {1, 2, 3}:
            raise ValueError(
                "Only spatial_dim = [1, 2, 3] is supported, "
                f"but got spatial_dim={spatial_dim}"
            )
        if kernel % 2 != 1:
            raise ValueError("Kernel size must be odd to achieve same-padding.")

        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.kernel = kernel
        self.stride = stride or 1
        self.padding = padding or kernel // 2  # Use same-padding by default
        self.dilation = dilation or 1
        self.groups = groups or 1
        self.bias = bias
        self.transposed = transposed
        self.volterra_low_rank = volterra_low_rank
        self.volterra_rank = volterra_rank

        if not volterra_low_rank:
            # Exact 2nd order Volterra kernel
            if spatial_dim != 1 or transposed:
                raise ValueError(
                    "Exact 2nd-order Volterra filter is only implemented for "
                    "dim=1 and transposed=False."
                )

            self.conv_order1 = getattr(nn, f"Conv{spatial_dim}d")(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )

            self.conv_order2 = getattr(nn, f"Conv{2*spatial_dim}d")(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
        else:
            if volterra_rank is None:
                raise ValueError(
                    "volterra_rank must be specified for low-rank approximation."
                )

            # Only set `output_padding` if it is given.
            additional_args = {}
            if output_padding is not None and transposed:
                additional_args["output_padding"] = output_padding

            # 2 * volterra_rank is for the low rank approximation of the 2nd order kernel
            # 1 is for the 1st order kernel
            self.conv = getattr(
                nn, f"Conv{'Transpose' if transposed else ''}{spatial_dim}d"
            )(
                in_channels=in_channels * (2 * volterra_rank + 1),
                out_channels=out_channels * (2 * volterra_rank + 1),
                kernel_size=kernel,
                stride=stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * (2 * volterra_rank + 1),
                bias=self.bias,
                **additional_args,
            )

            # Linear layer for aggregating the product terms of the low-rank approximation
            self.low_ranks_mixer = nn.Linear(out_channels * volterra_rank, out_channels)

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
            - If out_size is provided, the output will be resized to:
                [batch_size, out_channels, *out_size]
        """
        if not self.volterra_low_rank:
            # First order convolution
            z_order1 = self.conv_order1(x)

            # Exact 2nd order Volterra convolution
            x_outer_prod = torch.einsum("bc..., bc... -> bc... ...", x, x)
            z_order2 = self.conv_order2(x_outer_prod)
            z_order2 = torch.diagonal(z_order2, dim1=-1, dim2=-2)
        else:
            # Low-rank approximation computation. volterra_low_rank=True
            # guarantees volterra_rank was validated non-None in __init__.
            assert self.volterra_rank is not None
            x = einops.repeat(x, "b c ... -> b (r c) ...", r=2 * self.volterra_rank + 1)
            z = self.conv(x)

            # Split into 1st order and 2nd order terms
            z_order1 = z[:, : self.out_channels, ...]
            z_order2 = z[:, self.out_channels :, ...]

            # Split and elementwise multiply
            z_order2_1, z_order2_2 = torch.chunk(z_order2, chunks=2, dim=1)
            z_order2 = z_order2_1 * z_order2_2

            # aggregate the product terms
            z_order2 = einops.rearrange(z_order2, "b c ... -> b ... c")
            z_order2 = self.low_ranks_mixer(z_order2)
            z_order2 = einops.rearrange(z_order2, "b ... c -> b c ...")

        out = z_order1 + z_order2

        # Resize output if requested
        if out_size is not None:
            out = self.interpolator(out, out_size)

        return out
