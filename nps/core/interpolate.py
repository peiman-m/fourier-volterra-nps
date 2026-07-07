from typing import Any

import torch
import torch.nn as nn


class Interpolator(nn.Module):
    """An interpolator class for different tensor dimensions.

    Handles interpolation for tensors with different spatial
    dimensions (1D, 2D, 3D) using appropriate interpolation methods
    and parameters.
    
    Args:
        mode: Interpolation mode override. If None, uses dimension-appropriate defaults.
        align_corners: Whether to align corners for interpolation.
        antialias: Whether to use antialiasing (only honored for the bilinear/
            bicubic modes; ignored otherwise, as torch supports it only there).
    """

    # Class-level default parameters for better performance
    _DEFAULT_PARAMS: dict[int, dict[str, Any]] = {
        1: {"mode": "linear", "align_corners": True, "antialias": False},
        2: {"mode": "bicubic", "align_corners": True, "antialias": True},
        3: {"mode": "trilinear", "align_corners": True, "antialias": False},
    }

    def __init__(
        self,
        mode: str | None = None,
        align_corners: bool | None = None,
        antialias: bool | None = None,
    ):
        """Initialize the interpolator with configurable parameters."""
        super().__init__()
        
        self._mode_override = mode
        self._align_corners_override = align_corners
        self._antialias_override = antialias

    def _get_interp_params(self, spatial_dim: int) -> dict[str, Any]:
        """Get interpolation parameters for the given spatial dimension."""
        if spatial_dim not in self._DEFAULT_PARAMS:
            raise ValueError(
                f"Unsupported dimension: {spatial_dim}. "
                f"Supported dimensions are: {list(self._DEFAULT_PARAMS.keys())}"
            )
        
        # Start with defaults
        params = self._DEFAULT_PARAMS[spatial_dim].copy()
        
        # Apply overrides if provided
        if self._mode_override is not None:
            params["mode"] = self._mode_override
        if self._align_corners_override is not None:
            params["align_corners"] = self._align_corners_override
        if self._antialias_override is not None and params["mode"] in (
            "bilinear",
            "bicubic",
        ):
            # torch supports antialias only for the bilinear/bicubic kernels;
            # ignore the override for other modes rather than letting
            # interpolate raise.
            params["antialias"] = self._antialias_override

        return params

    def forward(
        self,
        tensor: torch.Tensor,
        out_size: int | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Interpolate the input tensor to the target size.

        Args:
            tensor: Input tensor of shape [B, C, *spatial_shape]
            out_size: Target size for spatial dimensions.
                Can be an int or a tuple of ints.

        Returns:
            Interpolated tensor

        Raises:
            ValueError: If the input tensor's spatial dimension is not supported
                       or out_size dimensions don't match spatial dimensions.
        """
        if out_size is None:
            return tensor

        # Hydra's ``_convert_="all"`` hands YAML sequences in as plain
        # ``list``; promote to ``tuple`` so ``current_size == out_size``
        # works correctly (tuple == list is always False).
        if isinstance(out_size, list):
            out_size = tuple(out_size)

        B, C, *spatial_shape = tensor.shape
        spatial_dim = len(spatial_shape)

        # Convert single int to a tuple of the appropriate length
        if isinstance(out_size, int):
            out_size = (out_size,) * spatial_dim
        elif len(out_size) != spatial_dim:
            raise ValueError(
                f"out_size must have length {spatial_dim}, "
                f"but got {len(out_size)}"
            )

        # Check if resizing is needed
        current_size = tuple(spatial_shape)
        if current_size == out_size:
            return tensor

        # Each spatial rank interpolates on its native tensor shape: 1D linear
        # on [B, C, W], 2D bicubic on [B, C, H, W], 3D trilinear on [B, C, D, H, W].
        interp_params = self._get_interp_params(spatial_dim)

        return torch.nn.functional.interpolate(
            tensor, size=out_size, **interp_params
        )


class UpSamplingNd(nn.Module):
    """Up-sampling layer.

    Args:
        spatial_dim (int): Dimensionality.
        size (int, optional): Up-sampling factor. Defaults to `2`.
        interp_method (str, optional): Interpolation method. Can be set to "bilinear".
            Defaults to "nearest'.
    """

    def __init__(
        self,
        spatial_dim: int,
        size: int = 2,
        interp_method: str = "bilinear",
    ):
        super().__init__()

        self.layer = getattr(nn, "Upsample")(
            # `scale_factor` is applied to each dimension automatically:
            # it doesn't need to be repeated.
            scale_factor=size,
            mode=interp_method,
        )

    def forward(self, x):
        return self.layer(x)


class AvgPoolNd(nn.Module):
    """Average pooling layer.

    Args:
        dim (int): Dimensionality.
        kernel (int): Kernel size.
        stride (int, optional): Stride.
    """

    def __init__(
        self,
        spatial_dim: int,
        kernel: int,
        stride: None | int = None,
    ):
        super().__init__()

        self.layer = getattr(nn, f"AvgPool{spatial_dim}d")(
            kernel_size=kernel,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        return self.layer(x)
