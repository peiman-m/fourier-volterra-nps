from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn

from ...utils.group_actions import translation
from .base import BaseConvolution


class BaseSetConv(BaseConvolution, ABC):
    """Abstract base class for set convolution layers.
    Each channel of the output has a separate lengthscale parameter.
    """

    def __init__(
        self,
        *,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        divide_by_density: bool = True,
        epsilon: float | None = 1e-4,
    ) -> None:
        """Initialize base set convolution layer.

        Args:
            spatial_dim: Spatial dimensionality of input coordinates
            in_channels: Number of input feature channels
            divide_by_density: Whether to normalize by local density
            epsilon: Small constant to prevent division by zero
        """
        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        if spatial_dim <= 0:
            raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if epsilon is not None and epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")

        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.divide_by_density = divide_by_density
        self.epsilon = epsilon or 0.0

    def _process_output(
        self,
        zq_updated: torch.Tensor,
        z_density: torch.Tensor,
        zq: torch.Tensor | None = None,
        channels_dim: int = -1,
    ) -> torch.Tensor:
        """Process kernel regression output with optional normalization
        and residual connection.

        Args:
            zq_updated: Updated feature values
            z_density: Local density estimates
            zq: Optional residual features for skip connection
            channels_dim: Dimension along which to concatenate density and features

        Returns:
            Concatenated tensor of density and normalized features
        """
        if self.divide_by_density:
            zq_updated = zq_updated / (z_density + self.epsilon)

        if zq is not None:
            if zq.shape != zq_updated.shape:
                raise ValueError(
                    f"Residual shape {zq.shape} doesn't match "
                    f"updated shape {zq_updated.shape}"
                )
            zq_updated = zq_updated + zq

        return torch.cat((z_density, zq_updated), dim=channels_dim)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class SetConv(BaseSetConv):
    """Set convolution layer for kernel embedding on sets
    and vice versa."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        in_channels: int,
        init_lengthscale: float = 0.1,
        learnable_lengthscale: bool = True,
        divide_by_density: bool = True,
        epsilon: float | None = 1e-4,
    ) -> None:
        """Initialize off-grid set convolution layer.

        Args:
            spatial_dim: Input dimensionality of coordinates
            in_channels: Number of input feature channels
            init_lengthscale: Initial kernel lengthscale parameter
            learnable_lengthscale: Whether lengthscale is trainable
            divide_by_density: Whether to normalize output by density
            epsilon: Numerical stabilizer to prevent division by zero
        """
        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=2 * in_channels,
            divide_by_density=divide_by_density,
            epsilon=epsilon,
        )

        # Initialize learnable lengthscale parameters
        self._init_lengthscale(init_lengthscale, learnable_lengthscale)

    def _init_lengthscale(
        self,
        init_lengthscale: float,
        learnable_lengthscale: bool = True,
    ) -> None:
        """Initialize Gaussian kernel lengthscale parameters."""
        lengthscale_channels = self.in_channels
        init_tensor = torch.full(
            (self.spatial_dim, lengthscale_channels), init_lengthscale
        )

        # Use log-space parameterization for numerical stability
        self.lengthscale_param = nn.Parameter(
            torch.log(torch.exp(init_tensor) - 1.0),
            requires_grad=learnable_lengthscale,
        )

    @property
    def lengthscale(self) -> torch.Tensor:
        """Returns the positive lengthscale using softplus."""
        return 1e-5 + torch.nn.functional.softplus(self.lengthscale_param)

    def _compute_weights(
        self,
        *,
        xq: torch.Tensor,
        xkv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute kernel weights between query and key-value points.

        Args:
            xq: Query coordinates of shape (..., M, dim)
            xkv: Key-value coordinates of shape (..., N, dim)

        Returns:
            Kernel weights of shape (..., M, N, param_channels)
        """
        # Compute pairwise distances: (..., M, N, dim)
        diff = translation(xq, xkv)

        # Expand for per-channel lengthscales: (..., M, N, dim, 1)
        diff_expanded = diff[..., None]
        lengthscale_expanded = einops.repeat(self.lengthscale, "... c -> ... (2 c)")

        # Scale by lengthscale parameters
        scaled_diff = diff_expanded / lengthscale_expanded

        # (..., M, N, param_channels)
        squared_dist = torch.sum(scaled_diff**2, dim=-2)
        return torch.exp(-0.5 * squared_dist)

    def forward(
        self,
        *,
        xkv: torch.Tensor,
        xq: torch.Tensor,
        zv: torch.Tensor,
        zq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform set convolution from (xkv, zv) to xq.

        Args:
            xkv: Input coordinates of shape (..., N, dim)
            xq: Output coordinates of shape (..., M, dim)
            zv: Input values of shape (..., N, in_channels)
            zq: Optional residual values of shape (..., M, in_channels)

        Returns:
            Output tensor of shape (..., M, 2 * in_channels) containing
            concatenated density and value channels.
        """
        # Compute kernel weights: (..., M, N, 2*in_channels)
        weights = self._compute_weights(xq=xq, xkv=xkv)

        # Create augmented features with density channel
        density = torch.ones_like(zv)
        z_aug = torch.cat([density, zv], dim=-1)

        # Apply weights: (..., M, N, 2*in_channels)
        weighted_z = weights * z_aug[:, None, :, :]

        # Aggregate over input points: (..., M, 2*in_channels)
        agg = weighted_z.sum(dim=-2)

        # Split into density and feature channels
        z_density, zq_updated = agg.chunk(2, dim=-1)

        return self._process_output(zq_updated, z_density, zq, channels_dim=-1)


class GridSetConv(BaseSetConv):
    """Set convolution for regular grids using standard CNN operations."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        in_channels: int,
        convnet: nn.Module | None = None,
        divide_by_density: bool = True,
        epsilon: float | None = 1e-4,
    ) -> None:
        """Initialize on-grid set convolution layer.

        Args:
            spatial_dim: Input coordinate dimensionality
            in_channels: Number of input feature channels
            convnet: CNN for processing grid features (defaults to Identity)
            divide_by_density: Whether to normalize output by density
            epsilon: Numerical stabilizer for division by zero
        """
        base_net = convnet or nn.Identity()
        convnet_features = base_net
        convnet_density = base_net

        # Step 2: Determine out_channels
        if hasattr(base_net, "out_channels"):
            out_channels = 2 * getattr(base_net, "out_channels")
        elif isinstance(base_net, nn.Identity):
            out_channels = 2 * in_channels
        else:
            raise ValueError(
                "convnet must have 'out_channels' attribute or be None/Identity."
            )

        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            divide_by_density=divide_by_density,
            epsilon=epsilon,
        )

        self.convnet_features = convnet_features
        self.convnet_density = convnet_density

    def forward(
        self,
        *,
        xkv_mask: torch.Tensor,
        zv: torch.Tensor,
        zq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform grid-based set convolution.

        Args:
            xkv_mask: Input mask of shape (B, ...) or (B, C, ...)
            zv: Input values of shape (B, in_channels, ...)
            zq: Optional residual values of shape (B, in_channels, ...)

        Returns:
            Output tensor of shape (B, 2 * in_channels, ...) containing
            concatenated density and value channels.
        """
        # Ensure mask has channel dimension
        if xkv_mask.ndim < zv.ndim:
            xkv_mask = xkv_mask[:, None]  # Add channel dim

        # Validate tensor dimensions
        if xkv_mask.ndim != zv.ndim:
            raise ValueError(
                "Mask and features must have same ndim, "
                f"got {xkv_mask.ndim} vs {zv.ndim}"
            )

        # Validate channel dimensions
        if xkv_mask.shape[1] not in {1, zv.shape[1]}:
            raise ValueError(
                f"Mask channels ({xkv_mask.shape[1]}) must be 1 or match "
                f"feature channels ({zv.shape[1]})"
            )

        # Validate spatial dimensions
        if xkv_mask.shape[2:] != zv.shape[2:]:
            raise ValueError(
                f"Mask spatial dims {xkv_mask.shape[2:]} must match "
                f"feature spatial dims {zv.shape[2:]}"
            )

        # Apply convnets
        zq_updated = self.convnet_features(zv * xkv_mask)
        z_density = self.convnet_density(xkv_mask.float())

        return self._process_output(zq_updated, z_density, zq, channels_dim=1)
