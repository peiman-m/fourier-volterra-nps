import math
from abc import abstractmethod
from collections.abc import Sequence

import einops
import torch
from torch import nn

from .base import BaseConvolution


class SetFourierConvBase(BaseConvolution):
    """
    Abstract base class for set-based Fourier convolution layers.

    Provides shared infrastructure: frequency grid construction, Gaussian kernel
    computation with caching, forward/inverse Fourier transforms, and grouped
    channel handling. Subclasses implement `_build_kernel` and `forward`.

    Args:
        spatial_dim (int): Spatial dimension (e.g., 1 for 1D, 2 for 2D).
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        max_freq (float | tuple): Maximum frequency per dimension.
        freq_resolution (float | tuple): Frequency grid resolution per dimension.
        groups (int | None): Number of groups to divide channels. Defaults to 1.
        init_lengthscale (float): Initial lengthscale for the Gaussian kernel.
        learnable_lengthscale (bool): Whether the lengthscale is trainable.
        legnthscale_cache_tolerance (float): Tolerance for lengthscale caching.
        low_rank (bool): Whether to use low-rank factorization of complex kernel weights.
        rank (int | None): Rank for low-rank factorization.
        input_feature_mixing (bool): Whether to apply linear mixing to input features.
        output_feature_mixing (bool): Whether to apply linear mixing to output features.
    """

    # Registered buffers (declared for static typing; created in __init__).
    pos_half_freq_grid: torch.Tensor
    freq_volume_element: torch.Tensor

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        max_freq: float | tuple[float, ...],
        freq_resolution: float | tuple[float, ...],
        groups: int | None = None,
        init_lengthscale: float = 0.1,
        learnable_lengthscale: bool = True,
        legnthscale_cache_tolerance: float = 1e-5,
        low_rank: bool = False,
        rank: int | None = None,
        input_feature_mixing: bool = False,
        output_feature_mixing: bool = False,
    ) -> None:
        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.groups = groups if groups is not None else 1
        self.low_rank = low_rank
        self.rank = rank
        self.input_feature_mixing = input_feature_mixing
        self.output_feature_mixing = output_feature_mixing

        # Validate group divisibility
        if (in_channels % self.groups) != 0 or (out_channels % self.groups) != 0:
            raise ValueError(
                f"in_channels ({in_channels}) and out_channels ({out_channels}) "
                f"must each be divisible by groups ({self.groups})."
            )
        # Number of feature-channels per group (excluding density)
        self.in_channels_per_group = self.in_channels // self.groups
        self.out_channels_per_group = self.out_channels // self.groups

        # Effective input channels include density channel
        self.effective_in_channels_per_group = 2 * self.in_channels_per_group

        # Validate low-rank parameters
        if low_rank:
            if rank is None or not (
                1
                <= rank
                <= min(self.effective_in_channels_per_group, self.out_channels_per_group)
            ):
                raise ValueError(
                    f"For low_rank=True, rank must satisfy "
                    f"1 <= rank <= min(effective_in_channels_per_group, out_channels_per_group), "
                    f"got rank={rank}, "
                    f"effective_in_channels_per_group={self.effective_in_channels_per_group}, "
                    f"out_channels_per_group={self.out_channels_per_group}"
                )

        # Normalize inputs to construct frequency grid and quadrant slices
        max_freq = self._normalize_freq_param(max_freq, "max_freq")
        freq_resolution = self._normalize_freq_param(freq_resolution, "freq_resolution")
        self._construct_freq_grid(max_freq, freq_resolution)
        self._construct_quad_slices()

        # Initialize learnable lengthscale parameters
        self._init_lengthscale(init_lengthscale, learnable_lengthscale)

        # Initialize caching for Gaussian FT
        self.legnthscale_cache_tolerance = legnthscale_cache_tolerance
        self._cached_gaussian_ft = None
        self._cached_lengthscale = None

        # Initialize Fourier domain kernels (subclass-defined)
        self._build_kernel()

        # Initialize optional feature mixing layers
        if self.input_feature_mixing:
            self.input_mixing_layer = nn.Linear(in_channels, in_channels)
        else:
            self.input_mixing_layer = None

        if self.output_feature_mixing:
            self.output_mixing_layer = nn.Linear(out_channels, out_channels)
        else:
            self.output_mixing_layer = None

    def _normalize_freq_param(
        self, param: float | tuple[float, ...], param_name: str
    ) -> tuple[float, ...]:
        """Normalize frequency parameter to tuple format."""
        if isinstance(param, (float, int)):
            if param <= 0:
                raise ValueError(f"{param_name} must be positive")
            return (float(param),) * self.spatial_dim

        elif isinstance(param, Sequence):
            if len(param) != self.spatial_dim:
                raise ValueError(
                    f"{param_name} must have length {self.spatial_dim}, "
                    f"got {len(param)}"
                )
            if not all(f > 0 for f in param):
                raise ValueError(f"All values in {param_name} must be positive")
            return tuple(float(f) for f in param)

        else:
            raise TypeError(
                f"{param_name} must be a positive float or tuple of positive floats"
            )

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

    def _construct_freq_grid(
        self,
        max_freq: tuple[float, ...],
        freq_resolution: tuple[float, ...],
    ) -> None:
        """Create frequency coordinate grid buffer."""
        pos_half_grid_axes: list[torch.Tensor] = []

        for i, (axis_max, axis_res) in enumerate(zip(max_freq, freq_resolution)):
            n_points = int(axis_max / axis_res)
            if n_points <= 0:
                raise ValueError(
                    f"Invalid grid size {n_points} for dimension {i}. "
                    f"Check max_freq={axis_max} and freq_resolution={axis_res}"
                )
            pos_freqs = torch.arange(0, n_points) * axis_res

            # For all dimensions except the last, include negative frequencies
            if i < self.spatial_dim - 1:
                neg_freqs = -torch.flip(pos_freqs[1:], dims=[0])
                axis_freqs = torch.cat([neg_freqs, pos_freqs], dim=0)
                pos_half_grid_axes.append(axis_freqs)
            else:
                # Last dimension: only non-negative frequencies
                pos_half_grid_axes.append(pos_freqs)

        pos_half_grid_meshes = torch.meshgrid(*pos_half_grid_axes, indexing="ij")
        pos_half_freq_grid = torch.stack(
            pos_half_grid_meshes, dim=-1
        )  # Shape: [f1, ..., fd, d]

        self.register_buffer("pos_half_freq_grid", pos_half_freq_grid)

        # Compute frequency volume element as product of axis resolutions
        freq_volume_element = math.prod(freq_resolution)
        self.register_buffer("freq_volume_element", torch.tensor(freq_volume_element))

    def _construct_quad_slices(self) -> None:
        """
        Construct slices for frequency quadrants.

        Due to Hermitian symmetry, we only store positive frequencies in the last
        dimension. For other dimensions, we have both positive and negative frequencies,
        creating 2^(d-1) quadrants that we need to handle separately.

        Each quadrant has equal size (mid = grid_size // 2 + 1), with overlapping
        at the center to ensure symmetric coverage.
        """
        # Get grid shape (excluding the final dimension coordinate)
        grid_shape = self.pos_half_freq_grid.shape[:-1]

        # Number of quadrants is 2^(spatial_dim - 1)
        num_quads = 2 ** (self.spatial_dim - 1)

        quad_slices = []
        for q in range(num_quads):
            slices = [slice(None)]  # Batch dimension

            # For each spatial dimension except the last
            for d in range(self.spatial_dim - 1):
                mid = grid_shape[d] // 2 + 1
                # Check if bit d is set in quadrant number q
                if (q >> d) & 1:
                    slices.append(slice(-mid, None))  # Positive half (last mid elements)
                else:
                    slices.append(slice(0, mid))  # Negative half (first mid elements)

            # Add ellipsis for remaining dimensions:
            #   1.Last dimension is always the full range (only positive frequencies)
            #   2.Feature channel dimension
            quad_slices.append(tuple(slices) + (...,))

        self.quad_slices = quad_slices

    @abstractmethod
    def _build_kernel(self) -> None:
        """Initialize learnable Fourier domain kernel weights."""

    def _compute_gaussian_ft(self) -> torch.Tensor:
        """
        Compute the fourier transform of gaussian kernel over the grid.
        Uses caching to avoid recomputation when lengthscale is unchanged.
        """
        # Check if we can use cached result
        if (
            self._cached_gaussian_ft is not None
            and self._cached_lengthscale is not None
            and torch.allclose(
                self.lengthscale,
                self._cached_lengthscale,
                atol=self.legnthscale_cache_tolerance,
                rtol=0.0,
            )
        ):
            return self._cached_gaussian_ft

        # Compute Gaussian FT
        freq = self.pos_half_freq_grid[
            ..., None
        ]  # add channel dimensions -> shape [nf_1, nf_2, .., nf_d, d, 1]
        ls = self.lengthscale.view(
            (1,) * self.spatial_dim + self.lengthscale.shape
        )  # broadcast to freq shape
        freq_sq = ((freq * ls) ** 2).sum(dim=-2)  # sum over dimensions
        cov_det_sqrt = torch.prod(self.lengthscale, dim=0, keepdim=True)
        gaussian_ft = (
            (2 * torch.pi) ** (self.spatial_dim / 2)
            * cov_det_sqrt
            * torch.exp(-2 * (torch.pi**2) * freq_sq)
        )  # shape: [nf_1, nf_2, .., nf_d, in_channels]

        # Cache the result
        self._cached_gaussian_ft = gaussian_ft.detach().clone()
        self._cached_lengthscale = self.lengthscale.detach().clone()

        return gaussian_ft

    def compute_translation_operands(self, xkv: torch.Tensor) -> torch.Tensor:
        """
        Precompute forward translation operators for given positions.

        Args:
            xkv: Input positions [B, N_kv, d]

        Returns:
            Precomputed translation operators [B, N_kv, f1, ..., fd]
        """
        # [B, N_kv, d] × [f1, ..., fd, d] → [B, N_kv, f1, ..., fd]
        phases = torch.einsum("bnd, ...d -> bn...", xkv, self.pos_half_freq_grid)
        return torch.exp(-2j * torch.pi * phases).to(torch.complex64)

    def compute_ift_operands(self, xq: torch.Tensor) -> torch.Tensor:
        """
        Precompute inverse Fourier transform operators for given positions.

        Args:
            xq: Query positions [B, N_q, d]

        Returns:
            Precomputed IFT operators [B, N_q, f1, ..., f_{d-1}, f_d]
        """
        # [B, N_q, d] × [f1, ..., f_{d-1}, f_d, d] → [B, N_q, f1, ..., f_{d-1}, f_d]
        ift_phases = torch.einsum("bnd ,...d -> bn...", xq, self.pos_half_freq_grid)
        return torch.exp(2j * torch.pi * ift_phases).to(torch.complex64)

    def _to_group_format(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return einops.rearrange(tensor, "... (g c) -> ... g c", g=self.groups)

    def _from_group_format(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return einops.rearrange(tensor, "... g c -> ... (g c)")

    def _forward_fourier(
        self,
        zv: torch.Tensor,
        xkv: torch.Tensor,
        precomputed_translation_ft: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the Fourier transform of the functional embedding.
        """
        # Use precomputed translation operators if provided
        if precomputed_translation_ft is not None:
            translation_ft = precomputed_translation_ft
        else:
            # Fallback to computing on-the-fly
            translation_ft = self.compute_translation_operands(xkv)

        # Density: sum of translation operators, shape [B, f1, .., fd]
        # Keep as [B, f1, .., fd, 1] to broadcast against gaussian [1, f1, .., fd, C_in]
        density_ft = translation_ft.sum(dim=1)[..., None]  # [B, f1, .., fd, 1]

        feature_ft = torch.einsum(
            "bnc, bn... -> b...c", zv.to(dtype=torch.complex64), translation_ft
        )  # [B, f1, .., fd, in_channels]

        # Apply Gaussian envelope to each part separately — avoids materialising a
        # C_in-fold density repeat and a 2×-expanded gaussian copy on every forward pass.
        gaussian_ft = self._compute_gaussian_ft()[None]  # [1, f1, .., fd, in_channels]
        return torch.cat(
            [density_ft * gaussian_ft, feature_ft * gaussian_ft], dim=-1
        )  # [B, f1, .., fd, 2*in_channels]

    def _inverse_fourier(
        self,
        z_ft: torch.Tensor,
        xq: torch.Tensor,
        precomputed_ift_operands: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute inverse Fourier transform using only positive half-space frequencies.

        For real-valued functions with Hermitian symmetric Fourier transforms:
        g(x) = 2·∫_H [Re(ĝ)·cos(2πξ·x) - Im(ĝ)·sin(2πξ·x)] dξ

        where H is the positive half-space (last coordinate ≥ 0). The boundary
        (ξ_d = 0) should only be counted once, so we rescale boundary coefficients
        by 0.5 before doubling.

        Args:
            z_ft: Fourier coefficients on positive half-space [B, f1, ..., fd, C]
            xq: Query positions [B, N_q, d]

        Returns:
            Real-valued function evaluations [B, N_q, C]
        """
        # Rescale boundary coefficients (ξ_d = 0) by 0.5 in-place.
        # z_ft is always a freshly allocated local at the call site and is never
        # read again after this method returns, so in-place mutation is safe.
        z_ft[..., 0, :] *= 0.5

        # Extract real and imaginary parts
        z_ft_real = z_ft.real  # [B, f1, ..., fd, C]
        z_ft_imag = z_ft.imag  # [B, f1, ..., fd, C]

        # Use precomputed IFT operators if provided
        if precomputed_ift_operands is not None:
            ift_operands = precomputed_ift_operands
        else:
            # Fallback to computing on-the-fly
            ift_operands = self.compute_ift_operands(xq)

        # Decompose into cosine and sine components
        cos_phases = ift_operands.real  # cos(2πξ·x), shape: [B, N_q, f1, ..., fd]
        sin_phases = ift_operands.imag  # sin(2πξ·x), shape: [B, N_q, f1, ..., fd]

        # Compute contribution with factor of 2 for symmetry
        real_term = torch.einsum("b...c, bn... -> bnc", z_ft_real, cos_phases)
        imag_term = torch.einsum("b...c, bn... -> bnc", z_ft_imag, sin_phases)
        result = 2.0 * (real_term - imag_term)

        # Scale by frequency volume element
        result = result * self.freq_volume_element

        return result

    @abstractmethod
    def forward(
        self,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        precomputed_translation_ft: torch.Tensor | None = None,
        precomputed_ift_operands: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform set convolution from (xkv, zv) to xq."""

    def to(self, *args, **kwargs):
        super_out = super().to(*args, **kwargs)
        # clear cache so it will rebuild on first use
        self._cached_lengthscale = None
        self._cached_gaussian_ft = None
        return super_out
