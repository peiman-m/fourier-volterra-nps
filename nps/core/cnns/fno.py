import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

from ...utils.helpers import compress_batch_dimensions
from ..convolution_blocks import FourierBlock, SequentialBlock
from ..mlp import MLP
from .base import BaseCNN


class FNO(BaseCNN):
    """Fourier Neural Operator.

    Args:
        spatial_dim (int): Dimensionality (1D or 2D).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (int | tuple[int, ...] | tuple[tuple[int, ...], ...]): Number of Fourier modes to keep.
            If an int, this number of modes is used for all dimensions and layers.
            If a flat tuple of ints with length `spatial_dim`, these modes are used per dimension
            for all layers. A flat int tuple is always interpreted as per-dimension, never per-layer.
            If a tuple of tuples (or mixed int/tuple), each element specifies modes for one layer:
            an int element applies to all dimensions, a tuple element specifies modes per dimension.
        groups (int | tuple[int, ...] | None): Number of groups for grouped spectral convolution.
            If None, all convolutions use groups=1 (standard convolution).
            If an int, this groups number is used for all layers.
            If tuple, specifies the groups number for each layer individually.
        channels (int | tuple[int, ...]): Number of channels at every intermediate layer.
            If int, creates `num_layers` layers with the same channel count.
            If tuple, specifies the channel count for each layer individually.
        num_layers (int | None): Number of layers. Used if `channels` is an int.
        activation (nn.Module | None): Activation function to use. Defaults to nn.GELU().
        p_dropout (float): Dropout probability (default: 0.0).
        residual (bool): Use residual connections. Defaults to True.
        separable (bool): Use depthwise separable spectral convolutions. Defaults to False.
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        modes: int | tuple[int, ...] | tuple[tuple[int, ...]],
        channels: int | tuple[int, ...],
        groups: int | tuple[int, ...] | None = None,
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
        activation = nn.GELU() if activation is None else activation
        self.channels = self._process_channels(channels, num_layers)
        self.layer_modes = self._process_modes(
            spatial_dim, len(self.channels), modes
        )
        self.layer_groups = self._process_groups(len(self.channels), groups)

        # Build FNO layers
        layers = []

        # Input projection
        layers.append(
            MLP(
                in_dim=in_channels,
                out_dim=self.channels[0],
                feature_axis=1,
            )
        )

        for i, (in_ch, out_ch) in enumerate(zip(self.channels[:-1], self.channels[1:])):
            layers.append(
                FourierBlock(
                    spatial_dim=spatial_dim,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    modes=self.layer_modes[i],
                    groups=self.layer_groups[i],
                    activation=activation,
                    separable=separable,
                    residual=residual,
                )
            )
            if p_dropout > 0:
                layers.append(nn.Dropout(p_dropout))

        # Final spectral block
        layers.append(
            FourierBlock(
                spatial_dim=spatial_dim,
                in_channels=self.channels[-1],
                out_channels=self.channels[-1],
                modes=self.layer_modes[-1],
                groups=self.layer_groups[-1],
                activation=activation,
                separable=separable,
                residual=residual,
            )
        )

        # Output projection
        layers.append(
            MLP(
                in_dim=self.channels[-1],
                out_dim=out_channels,
                feature_axis=1,
            )
        )

        self.net = SequentialBlock(*layers)

    @staticmethod
    def _process_channels(channels, num_layers):
        if isinstance(channels, int):
            if num_layers is None:
                warnings.warn(
                    "Single int `channels` and no `num_layers` provided. Using 1 layer."
                )
            return (channels,) * (num_layers or 1)
        if isinstance(channels, Sequence):
            if num_layers is not None:
                warnings.warn("`channels` is a sequence. `num_layers` will be ignored.")
            return tuple(channels)
        raise TypeError("`channels` must be an int or a sequence of ints.")

    @staticmethod
    def _process_modes(spatial_dim, num_layers, modes):
        if isinstance(modes, int):
            return [(modes,) * spatial_dim] * num_layers

        if isinstance(modes, Sequence):
            # Flat tuple of ints → per-dimension, same for all layers.
            # Always takes this interpretation regardless of num_layers.
            if all(isinstance(m, int) for m in modes):
                if len(modes) != spatial_dim:
                    raise ValueError(
                        f"Flat `modes` tuple must have {spatial_dim} elements "
                        f"(one per spatial dimension), got {len(modes)}."
                    )
                return [tuple(modes)] * num_layers

            # Tuple of tuples (or mixed int/tuple) → per-layer specification.
            if len(modes) != num_layers:
                raise ValueError(
                    f"Per-layer `modes` must have {num_layers} elements, got {len(modes)}."
                )
            processed = []
            for m in modes:
                if isinstance(m, int):
                    processed.append((m,) * spatial_dim)
                elif len(m) == spatial_dim:
                    processed.append(tuple(m))
                else:
                    raise ValueError(
                        f"Each mode tuple must have {spatial_dim} elements, got {len(m)}."
                    )
            return processed

        raise TypeError(
            "`modes` must be int, flat tuple of ints (per dimension), "
            "or tuple of tuples (per layer)."
        )

    @staticmethod
    def _process_groups(num_layers, groups):
        if groups is None:
            return [None] * num_layers
        if isinstance(groups, int):
            return [groups] * num_layers
        if isinstance(groups, Sequence):
            if len(groups) == num_layers:
                return tuple(groups)
            raise ValueError(
                f"`groups` must have length {num_layers}, got {len(groups)}."
            )
        raise TypeError("`groups` must be None, an int, or a sequence of ints.")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z, uncompress = compress_batch_dimensions(z, self.spatial_dim + 1)
        z = self.net(z)
        return uncompress(z)
