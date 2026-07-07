import copy
import warnings
from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from .linear import GroupLinear


class MLP(nn.Module):
    """
    A flexible multi-layer perceptron (MLP) with optional embedding, batch normalization,
    dropout, and customizable non-linearities and weight initialization.

    Args:
        in_dim (int): Dimension of the input features.
        out_dim (int): Dimension of the output features.
        feature_axis (int): Dimension index to transform.
            Supports negative indexing. Default: -1.
        layer_dim (int | tuple[int, ...] | None): Sizes of the hidden layers.
        num_layers (int | None): Number of hidden layers (used if `layers` is int).
        activation (nn.Module | None): Non-linearity to use (default: nn.ReLU()).
        input_activation (nn.Module | None): Activation for the inputs (default: None).
        output_activation (nn.Module | None): Activation for the output layer (default: None).
        groups (int | None): Number of groups for grouped linear layers.
            If > 1, uses GroupLinear instead of Linear.
        batchnorm (bool): Whether to include batch normalization layers (default: False).
        p_dropout (float): Dropout probability (default: 0.0).
        embedding (nn.Module | None): Embedding module to process the input.
        init_fn (Callable[[nn.Module], None] | None): Function for custom weight initialization.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        feature_axis: int = -1,
        layer_dim: int | tuple[int, ...] | None = None,
        num_layers: int | None = None,
        activation: nn.Module | None = None,
        input_activation: nn.Module | None = None,
        output_activation: nn.Module | None = None,
        groups: int | None = None,
        bias: bool = True,
        batchnorm: bool = False,
        p_dropout: float = 0.0,
        embedding: nn.Module | None = None,
        init_fn: Callable[[nn.Module], None] | None = None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feature_axis = feature_axis
        self.groups = groups or 1

        # Determine layer sizes
        if layer_dim is not None:
            if isinstance(layer_dim, int):
                if num_layers is None:
                    warnings.warn(
                        "`layer_dim` is a single integer and `num_layers` is None."
                        f"This will create a single hidden layer of size {layer_dim}."
                    )
                layers = (layer_dim,) * (num_layers or 1)

            elif isinstance(layer_dim, list):
                if num_layers is not None:
                    warnings.warn(
                        "`layer_dim` is a list and `num_layers` is provided. "
                        "Ignoring `num_layers`."
                    )
                layers = layer_dim
        else:
            layers = ()

        # Build network
        self.net = self._build_network(
            layers,
            bias,
            batchnorm,
            p_dropout,
            embedding,
            activation,
            input_activation,
            output_activation,
        )

        # Custom weight initialization
        self._initialize_weights(init_fn)

    def _build_network(
        self,
        layers: tuple[int, ...],
        bias: bool,
        batchnorm: bool,
        p_dropout: float,
        embedding: nn.Module | None = None,
        activation: nn.Module | None = None,
        input_activation: nn.Module | None = None,
        output_activation: nn.Module | None = None,
    ) -> nn.Sequential:
        """Constructs the MLP network with specified configurations."""

        # Non-linear activation default
        activation = activation or nn.ReLU()

        modules = []

        # Embedding handling
        prev_dim = self._attach_embedding(modules, embedding)

        # Input activation
        if input_activation:
            modules.append(copy.deepcopy(input_activation))

        # Hidden layers
        for layer_dim in layers:
            if self.groups > 1:
                modules.append(
                    GroupLinear(prev_dim, layer_dim, groups=self.groups, bias=bias)
                )
            else:
                modules.append(nn.Linear(prev_dim, layer_dim, bias=bias))
            if batchnorm:
                modules.append(nn.BatchNorm1d(layer_dim))
            modules.append(copy.deepcopy(activation))
            if p_dropout > 0:
                modules.append(nn.Dropout(p_dropout))
            prev_dim = layer_dim

        # Final layer
        if self.groups > 1:
            modules.append(
                GroupLinear(prev_dim, self.out_dim, groups=self.groups, bias=bias)
            )
        else:
            modules.append(nn.Linear(prev_dim, self.out_dim, bias=bias))

        if output_activation:
            modules.append(copy.deepcopy(output_activation))

        return nn.Sequential(*modules)

    def _attach_embedding(
        self,
        modules: list,
        embedding: nn.Module | None = None,
    ) -> int:
        """Attach an optional input embedding and return the width fed to the
        first linear layer.

        ``in_dim`` is the *raw* input width the MLP's ``forward`` receives. When
        an embedding is attached it maps ``in_dim`` -> ``embedding.out_dim``, and
        the first linear layer consumes ``embedding.out_dim``. The embedding's
        own ``in_dim`` should therefore equal the MLP's ``in_dim``.
        """
        if embedding is not None:
            if getattr(embedding, "out_dim", None) is None:
                raise ValueError("Embedding module must have an `out_dim` attribute.")

            emb_in_dim = getattr(embedding, "in_dim", None)
            if emb_in_dim is not None and emb_in_dim != self.in_dim:
                warnings.warn(
                    f"Embedding in_dim ({emb_in_dim}) does not match MLP in_dim "
                    f"({self.in_dim}). MLP.forward validates raw inputs against "
                    "in_dim, so these should agree."
                )

            modules.append(embedding)
            # out_dim is an int (guarded above); cast away the
            # nn.Module.__getattr__ Tensor | Module leak.
            return cast(int, embedding.out_dim)

        return self.in_dim

    def _initialize_weights(self, init_fn: Callable[[nn.Module], None] | None = None):
        """Applies custom or default weight initialization."""
        if init_fn is not None:
            self.net.apply(init_fn)
        else:
            # Default weight initialization for Linear and GroupLinear layers
            def default_init(m):
                if isinstance(m, (nn.Linear, GroupLinear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

            self.net.apply(default_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: applies MLP over the specified dimension."""
        # Normalize dim to positive index
        feature_axis = (
            self.feature_axis if self.feature_axis >= 0 else x.ndim + self.feature_axis
        )
        if not (0 <= feature_axis < x.ndim):
            raise ValueError(
                f"Invalid feature_axis={self.feature_axis} "
                f"for input with {x.ndim} dims."
            )

        # Move target dim to last
        if feature_axis != x.ndim - 1:
            x = x.transpose(feature_axis, -1)

        # Check feature size
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.in_dim}, "
                f"got {x.shape[-1]}."
            )

        out = self.net(x)

        # Move output dim back to original position if needed
        if feature_axis != x.ndim - 1:
            out = out.transpose(feature_axis, -1)

        return out
