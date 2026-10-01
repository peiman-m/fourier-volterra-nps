import copy
from functools import partial
from typing import Callable, cast

import torch
import torch.nn as nn

from ...utils.helpers import compress_batch_dimensions
from ..convolution_blocks import ConvBlock
from ..convolutions import ConvNd
from ..interpolate import AvgPoolNd, UpSamplingNd
from .base import BaseCNN


class UNet(BaseCNN):
    """UNet.
    Reference: https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses/coders/nn.py

    Args:
        spatial_dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        kernels (int or tuple[int], optional): Sizes of the kernels. Defaults to `5`.
        strides (int or tuple[int], optional): Strides. Defaults to `2`.
        activations (object or tuple[object], optional): Activation functions.
        separable (bool, optional): Use depthwise separable convolutions. Defaults to
            `False`.
        residual (bool, optional): Make residual convolutional blocks. Defaults to
            `False`.
        resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions. Defaults to `False`.
        resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions. Can be set to "bilinear". Defaults to "nearest".

    Attributes:
        spatial_dim (int): Dimensionality.
        kernels (tuple[int]): Sizes of the kernels.
        strides (tuple[int]): Strides.
        activations (tuple[function]): Activation functions.
        num_halving_layers (int): Number of layers with stride equal to two.
        receptive_fields (list[float]): Receptive field for every intermediate value.
        receptive_field (float): Receptive field of the model.
        before_turn_layers (list[module]): Layers before the U-turn.
        after_turn_layers (list[module]): Layers after the U-turn
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] = (8, 16, 16, 32, 32, 64),
        kernels: int | tuple[int | tuple[int, ...], ...] = 5,
        strides: int | tuple[int, ...] = 2,
        activations: nn.Module | tuple[nn.Module, ...] | None = None,
        separable: bool = False,
        residual: bool = True,
        resize_convs: bool = False,
        resize_conv_interp_method: str = "nearest",
    ) -> None:
        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        # If `kernel` is an integer, repeat it for every layer.
        if not isinstance(kernels, (tuple, list)):
            kernels = (kernels,) * len(channels)
        elif len(kernels) != len(channels):
            raise ValueError(
                f"Length of `kernels` ({len(kernels)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.kernels = kernels

        # If `strides` is an integer, repeat it for every layer.
        # TODO: Change the default so that the first stride is 1.
        if not isinstance(strides, (tuple, list)):
            strides = (strides,) * len(channels)
        elif len(strides) != len(channels):
            raise ValueError(
                f"Length of `strides` ({len(strides)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.strides = strides

        # Default to ReLUs. Moreover, if `activations` is an activation function, repeat
        # it for every layer.
        activations = activations or nn.ReLU()
        if not isinstance(activations, (tuple, list)):
            activations = tuple(
                copy.deepcopy(activations) for _ in range(len(channels))
            )
        elif len(activations) != len(channels):
            raise ValueError(
                f"Length of `activations` ({len(activations)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.activations = activations

        # Compute number of halving layers.
        self.num_halving_layers = len(channels)

        # Compute receptive field at all stages of the model.
        self.receptive_fields = [1]
        # Forward pass:
        for stride, kernel in zip(self.strides, self.kernels):
            # Deal with composite kernels:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            after_conv = self.receptive_fields[-1] + (kernel - 1)
            if stride > 1:
                if after_conv % 2 == 0:
                    # If even, then subsample.
                    self.receptive_fields.append(after_conv // 2)
                else:
                    # If odd, then average pool.
                    self.receptive_fields.append((after_conv + 1) // 2)
            else:
                self.receptive_fields.append(after_conv)
        # Backward pass:
        for stride, kernel in zip(reversed(self.strides), reversed(self.kernels)):
            # Deal with composite kernels:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            if stride > 1:
                after_interp = self.receptive_fields[-1] * 2 - 1
                self.receptive_fields.append(after_interp + (kernel - 1))
            else:
                self.receptive_fields.append(self.receptive_fields[-1] + (kernel - 1))
        self.receptive_field = self.receptive_fields[-1]

        # If none of the fancy features are used, use the standard `ConvNd` for
        # compatibility with trained models. For the same reason we also don't use the
        #   `activation` keyword.
        # TODO: In the future, use `ConvBlock` everywhere and use the `activation`
        #   keyword.
        if residual or separable or any(isinstance(k, tuple) for k in kernels):
            Conv = cast(
                Callable[..., nn.Module],
                partial(
                    ConvBlock,
                    spatial_dim=spatial_dim,
                    residual=residual,
                    separable=separable,
                ),
            )
        else:

            def Conv(
                *, stride: int = 1, transposed: bool = False, **kw_args
            ) -> nn.Module:
                if transposed and stride > 1:
                    kw_args["output_padding"] = stride // 2
                kw_args.pop("activation", None)
                return ConvNd(
                    spatial_dim=spatial_dim,
                    stride=stride,
                    transposed=transposed,
                    **kw_args,
                )

        def construct_before_turn_layer(i: int) -> nn.Module:
            # Determine the configuration of the layer.
            ci = ((in_channels,) + tuple(channels))[i]
            co = channels[i]
            k = self.kernels[i]
            s = self.strides[i]
            act = self.activations[i]

            if s == 1:
                # Just a regular convolutional layer.
                return Conv(
                    in_channels=ci,
                    out_channels=co,
                    kernel=k,
                    activation=act,
                )
            else:
                # This is a downsampling layer.
                if self.receptive_fields[i] % 2 == 1:
                    # Perform average pooling if the previous receptive field is odd.
                    return nn.Sequential(
                        Conv(
                            in_channels=ci,
                            out_channels=co,
                            kernel=k,
                            stride=1,
                            activation=act,
                        ),
                        AvgPoolNd(
                            spatial_dim=spatial_dim,
                            kernel=s,
                            stride=s,
                        ),
                    )
                else:
                    # Perform subsampling if the previous receptive field is even.
                    return Conv(
                        in_channels=ci,
                        out_channels=co,
                        kernel=k,
                        stride=s,
                        activation=act,
                    )

        def construct_after_turn_layer(i: int) -> nn.Module:
            # Determine the configuration of the layer.
            if i == len(channels) - 1:
                # No skip connection yet.
                ci = channels[i]
            else:
                # Add the skip connection.
                ci = 2 * channels[i]
            co = ((channels[0],) + tuple(channels))[i]
            k = self.kernels[i]
            s = self.strides[i]
            act = self.activations[i]

            if s == 1:
                # Just a regular convolutional layer.
                return Conv(
                    in_channels=ci,
                    out_channels=co,
                    kernel=k,
                    activation=act,
                )
            else:
                # This is an upsampling layer.
                if resize_convs:
                    return nn.Sequential(
                        UpSamplingNd(
                            spatial_dim=spatial_dim,
                            size=s,
                            interp_method=resize_conv_interp_method,
                        ),
                        Conv(
                            in_channels=ci,
                            out_channels=co,
                            kernel=k,
                            stride=1,
                            activation=act,
                        ),
                    )
                else:
                    return Conv(
                        in_channels=ci,
                        out_channels=co,
                        kernel=k,
                        stride=s,
                        transposed=True,
                        activation=act,
                    )

        self.before_turn_layers = nn.ModuleList(
            [construct_before_turn_layer(i) for i in range(len(channels))]
        )
        self.after_turn_layers = nn.ModuleList(
            [construct_after_turn_layer(i) for i in range(len(channels))]
        )
        self.final_linear = ConvNd(
            spatial_dim=spatial_dim,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, uncompress = compress_batch_dimensions(x, self.spatial_dim + 1)

        hs = [self.activations[0](self.before_turn_layers[0](x))]
        for layer, activation in zip(
            self.before_turn_layers[1:],
            self.activations[1:],
        ):
            hs.append(activation(layer(hs[-1])))

        # Now make the turn!

        h = self.activations[-1](self.after_turn_layers[-1](hs[-1]))
        for h_prev, layer, activation in zip(
            reversed(hs[:-1]),
            reversed(self.after_turn_layers[:-1]),
            reversed(self.activations[:-1]),
        ):
            h = activation(layer(torch.cat((h_prev, h), dim=1)))

        return uncompress(self.final_linear(h))
