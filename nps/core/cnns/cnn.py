import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

from ...utils.helpers import compress_batch_dimensions
from ..convolution_blocks import ConvBlock
from ..mlp import MLP
from .base import BaseCNN


class ConvNet(BaseCNN):
    """A regular convolutional neural network.

    Args:
        spatial_dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel (int | tuple[int, ...] | tuple[tuple[int, ...], ...]): Kernel size(s).
            If an int, this kernel size is used for all layers.
            If a tuple, must have the same length as `channels`.
        groups (int | tuple[int, ...] | None): Number of groups for grouped convolution.
            If an int, this group number is used for all layers.
            If tuple, specifies the groups number for each layer individually.
        channels (int | tuple[int, ...] | None): Number of channels at every intermediate layer.
            If int, creates `num_layers` layers with the same channel count.
            If tuple, specifies the channel count for each layer individually.
        num_layers (int | None): Number of layers. Used if `channels` is an int.
        activation (nn.Module | None): Activation function to use. Defaults to nn.ReLU().
        p_dropout (float): Dropout probability (default: 0.0).
        separable (bool): Use depthwise separable convolutions. Defaults to False.
        residual (bool): Use residual connections. Defaults to True.
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel: int | tuple[int, ...] | tuple[tuple[int, ...], ...],
        groups: int | tuple[int, ...] | None = None,
        channels: int | tuple[int, ...] | None = None,
        num_layers: int | None = None,
        activation: nn.Module | None = None,
        p_dropout: float = 0.0,
        separable: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        if channels is None:
            channels = (out_channels,)
        self.channels = self._process_channels(channels, num_layers)

        self.kernels = self._normalize_args(
            kernel, len(self.channels), "kernel", expected_type=int
        )
        self.groups_list = self._normalize_args(
            groups, len(self.channels), "groups", default=1, expected_type=int
        )
        self.p_dropout = p_dropout
        self.activation = activation or nn.ReLU()
        self.separable = separable
        self.residual = residual

        self.convnet = self._build_layers(
            in_channels,
            out_channels,
            self.channels,
            self.kernels,
            self.groups_list,
            self.activation,
            self.p_dropout,
            self.separable,
            self.residual,
        )

    @staticmethod
    def _process_channels(channels, num_layers):
        if isinstance(channels, int):
            if num_layers is None:
                warnings.warn(
                    "`channels` is an int and `num_layers` is None. "
                    "Proceed with a single layer."
                )
            return (channels,) * (num_layers or 1)
        elif isinstance(channels, Sequence):
            if num_layers is not None:
                warnings.warn("`channels` is a sequence. `num_layers` is ignored.")
            return tuple(channels)
        else:
            raise TypeError("`channels` must be int or sequence of ints.")

    @staticmethod
    def _normalize_args(
        value,
        expected_len,
        name,
        default=None,
        expected_type=None,
    ):
        if value is None:
            if default is not None:
                return (default,) * expected_len
            raise TypeError(f"`{name}` cannot be None.")
        if (
            expected_type
            and isinstance(value, expected_type)
            and not isinstance(value, Sequence)
        ):
            return (value,) * expected_len
        if isinstance(value, Sequence):
            if len(value) != expected_len:
                raise ValueError(
                    f"`{name}` must have {expected_len} elements, got {len(value)}."
                )
            if expected_type and not all(isinstance(v, expected_type) for v in value):
                raise TypeError(
                    f"All elements in `{name}` must be of type {expected_type.__name__}."
                )
            return tuple(value)
        raise TypeError(
            f"`{name}` must be "
            f"{expected_type.__name__ if expected_type else type(value).__name__}, "
            "sequence of such, or None."
        )

    def _build_layers(
        self,
        in_channels,
        out_channels,
        channels,
        kernels,
        groups,
        activation,
        p_dropout,
        separable,
        residual,
    ):
        layers = []
        prev_channels = in_channels

        for i, next_channels in enumerate(channels):
            layers.append(
                ConvBlock(
                    spatial_dim=self.spatial_dim,
                    in_channels=prev_channels,
                    out_channels=next_channels,
                    kernel=kernels[i],
                    stride=1,
                    groups=groups[i],
                    activation=activation,
                    separable=separable,
                    residual=residual,
                )
            )
            if p_dropout > 0:
                layers.append(nn.Dropout(p_dropout))
            prev_channels = next_channels

        # Output projection
        layers.append(
            MLP(
                in_dim=prev_channels,
                out_dim=out_channels,
                feature_axis=1,
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, uncompress = compress_batch_dimensions(x, self.spatial_dim + 1)
        return uncompress(self.convnet(x))
