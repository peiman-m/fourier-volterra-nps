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
        lengthscale_cache_tolerance (float): Tolerance for lengthscale caching.
        low_rank (bool): Whether to use low-rank factorization of complex kernel weights.
        rank (int | None): Rank for low-rank factorization.
        input_feature_mixing (bool): Whether to apply linear mixing to input features.
        output_feature_mixing (bool): Whether to apply linear mixing to output features.
        freq_chunks (int | None): Number of disjoint chunks along the first
            frequency axis for the spectral product. Purely a compute/memory
            schedule — results are identical for any value. None or 1 computes
            the full grid in one einsum. May be changed at runtime.
        density_feature_pairing (str): Channel layout handed to the grouped
            spectral kernel. "blocked" (default) concatenates
            [D-block | F-block]; for groups > 1 the contiguous group split
            hands some groups only density channels and others only feature
            channels. "interleaved" gives every group matched [D_g | F_g]
            pairs.
    """

    # Registered buffers (declared for static typing; created in __init__).
    pos_half_freq_grid: torch.Tensor
    ift_boundary_weights: torch.Tensor
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
        lengthscale_cache_tolerance: float = 1e-5,
        low_rank: bool = False,
        rank: int | None = None,
        input_feature_mixing: bool = False,
        output_feature_mixing: bool = False,
        freq_chunks: int | None = None,
        density_feature_pairing: str = "blocked",
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

        if density_feature_pairing not in ("interleaved", "blocked"):
            raise ValueError(
                f"density_feature_pairing must be 'interleaved' or 'blocked', "
                f"got {density_feature_pairing!r}"
            )
        self.density_feature_pairing = density_feature_pairing

        if freq_chunks is not None and freq_chunks < 1:
            raise ValueError(f"freq_chunks must be >= 1 or None, got {freq_chunks}")
        self.freq_chunks = freq_chunks

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

        # Normalize inputs to construct frequency grid
        max_freq = self._normalize_freq_param(max_freq, "max_freq")
        freq_resolution = self._normalize_freq_param(freq_resolution, "freq_resolution")
        self._construct_freq_grid(max_freq, freq_resolution)

        # Initialize learnable lengthscale parameters
        self._init_lengthscale(init_lengthscale, learnable_lengthscale)

        # Initialize caching for Gaussian FT
        self.lengthscale_cache_tolerance = lengthscale_cache_tolerance
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
        # bool is a subclass of int; reject it explicitly so e.g. max_freq=True
        # doesn't silently become 1.0.
        if isinstance(param, bool):
            raise TypeError(
                f"{param_name} must be a positive float or tuple of positive floats, "
                f"got bool"
            )

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
            if any(isinstance(f, bool) or not isinstance(f, (float, int)) for f in param):
                raise TypeError(
                    f"All values in {param_name} must be floats or ints"
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
            # round() avoids float-division truncation (e.g. int(0.3/0.1) == 2);
            # +1 so the grid spans [0, max_freq] inclusive.
            n_intervals = round(axis_max / axis_res)
            if n_intervals <= 0:
                raise ValueError(
                    f"Invalid grid size {n_intervals} for dimension {i}. "
                    f"Check max_freq={axis_max} and freq_resolution={axis_res}"
                )
            n_points = n_intervals + 1
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

        # Half-space IFT quadrature weights, [f1, ..., fd]: the xi_d = 0
        # boundary plane is counted once (0.5 before the global doubling in
        # `_inverse_fourier`), every other bin twice. Derived from the grid,
        # so not persisted in checkpoints.
        ift_boundary_weights = torch.where(
            pos_half_freq_grid[..., -1] == 0, 0.5, 1.0
        )
        self.register_buffer(
            "ift_boundary_weights", ift_boundary_weights, persistent=False
        )

        # Compute frequency volume element as product of axis resolutions
        freq_volume_element = math.prod(freq_resolution)
        self.register_buffer("freq_volume_element", torch.tensor(freq_volume_element))

    @abstractmethod
    def _build_kernel(self) -> None:
        """Initialize learnable Fourier domain kernel weights."""

    def _compute_gaussian_ft(self) -> torch.Tensor:
        """
        Compute the fourier transform of gaussian kernel over the grid.
        Uses caching to avoid recomputation when lengthscale is unchanged.
        """
        # The cache stores a detached tensor, so serving it while gradients are
        # live would cut the graph to lengthscale_param. Only cache when no
        # gradient can flow to the lengthscale (eval / no_grad / frozen).
        cache_allowed = not (
            torch.is_grad_enabled() and self.lengthscale_param.requires_grad
        )

        # Check if we can use cached result
        if (
            cache_allowed
            and self._cached_gaussian_ft is not None
            and self._cached_lengthscale is not None
            and torch.allclose(
                self.lengthscale,
                self._cached_lengthscale,
                atol=self.lengthscale_cache_tolerance,
                rtol=0.0,
            )
        ):
            return self._cached_gaussian_ft

        # Compute Gaussian FT
        freq = self.pos_half_freq_grid[
            ..., None
        ]  # add channel dimensions -> shape [nf_1, nf_2, .., nf_d, d, 1]
        # lengthscale [d, C] broadcasts right-aligned against freq [f..., d, 1]
        freq_sq = ((freq * self.lengthscale) ** 2).sum(dim=-2)  # -> [f..., C]
        cov_det_sqrt = torch.prod(self.lengthscale, dim=0, keepdim=True)
        gaussian_ft = (
            (2 * torch.pi) ** (self.spatial_dim / 2)
            * cov_det_sqrt
            * torch.exp(-2 * (torch.pi**2) * freq_sq)
        )  # shape: [nf_1, nf_2, .., nf_d, in_channels]

        # Cache the result
        if cache_allowed:
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

    def _spectral_product(self, embedding_ft: torch.Tensor) -> torch.Tensor:
        """
        Apply the Fourier-domain kernel to the grouped embedding, pointwise in
        frequency.

        Args:
            embedding_ft: Grouped embedding [B, f1, ..., fd, G, C_eff].

        Returns:
            Spectral output [B, f1, ..., fd, KG * out_channels_per_group],
            where KG is the kernel-group count (dim 0 of the kernel tensors):
            G for SetFourierConv, (2R+1)·G for the Volterra subclass.

        Kernel tensors may carry KG = blocks·G groups (r-major: k = r·G + g).
        The embedding is never repeated across blocks — the block axis is
        broadcast inside the einsum, so subclasses with blocks > 1 pay no
        blocks-fold copy of the embedding.

        When `freq_chunks` > 1, the product is computed over disjoint chunks of
        the flattened frequency grid to cap transient memory (einsum workspace
        and, in the low-rank case, the rank-projected intermediate). Chunking
        the flat bin index rather than any single axis keeps the granularity
        independent of the grid shape, so it works equally well for
        anisotropic per-axis max_freq / freq_resolution settings. The chunk
        count is a pure compute schedule: results are identical for any value,
        and the attribute may be changed at runtime.
        """
        # Kernel dim 0 fuses (blocks r, groups g), r-major: KG = r * G.
        # r = 1 for SetFourierConv; r = 2R+1 Volterra blocks for the subclass.
        kernel = self.U if self.low_rank else self.weights

        # Flatten the frequency axes on both operands (views for contiguous
        # tensors); `unpack` restores the grid shape on the way out.
        # [B, f1, .., fd, G, C_eff] -> [B, n_bins, G, C_eff]
        emb, freq_packing = einops.pack([embedding_ft], "b * g i")
        n_bins = emb.shape[1]  # total frequency bins, prod(f1..fd)
        if self.low_rank:
            # [KG, C_eff, rank, f1, .., fd] -> [r, G, C_eff, rank, n_bins]
            U = einops.rearrange(
                self.U, "(r g) i k ... -> r g i k (...)", g=self.groups
            )
            # [KG, rank, o, f1, .., fd] -> [r, G, rank, o, n_bins]
            V = einops.rearrange(
                self.V, "(r g) k o ... -> r g k o (...)", g=self.groups
            )
        else:
            # [KG, C_eff, o, f1, .., fd] -> [r, G, C_eff, o, n_bins]
            W = einops.rearrange(
                self.weights, "(r g) i o ... -> r g i o (...)", g=self.groups
            )

        def apply_kernel(emb_chunk: torch.Tensor, f_slc) -> torch.Tensor:
            """[B, f_chunk, G, C_eff] -> [B, f_chunk, KG * o]."""
            if self.low_rank:
                # Sequential contraction: avoids materialising W = U @ V.
                # Cost: O((C_eff + C_out) * rank * freq) vs O(C_eff * C_out * freq)
                projected = torch.einsum(
                    "bfgi, rgikf -> bfrgk", emb_chunk, U[..., f_slc]
                )
                out = torch.einsum(
                    "bfrgk, rgkof -> bfrgo", projected, V[..., f_slc]
                )
            else:
                out = torch.einsum("bfgi, rgiof -> bfrgo", emb_chunk, W[..., f_slc])
            # Flatten (r g o) r-major, matching the kernel-group channel layout
            return einops.rearrange(out, "b f r g o -> b f (r g o)")

        num_chunks = min(self.freq_chunks or 1, n_bins)
        if num_chunks <= 1:
            z_ft = apply_kernel(emb, slice(None))
        else:
            z_ft = torch.empty(
                emb.shape[0],  # batch B
                n_bins,  # flattened frequency grid
                kernel.shape[0] * self.out_channels_per_group,  # KG * o out channels
                dtype=emb.dtype,
                device=emb.device,
            )
            # Chunk boundaries over the flat bin index, e.g. n_bins=10 with
            # num_chunks=3 -> edges [0, 3, 7, 10]
            edges = [round(i * n_bins / num_chunks) for i in range(num_chunks + 1)]
            for start, end in zip(edges[:-1], edges[1:]):
                f_slc = slice(start, end)
                z_ft[:, f_slc] = apply_kernel(emb[:, f_slc], f_slc)

        # [B, n_bins, KG * o] -> [B, f1, .., fd, KG * o]
        [z_ft] = einops.unpack(z_ft, freq_packing, "b * c")
        return z_ft

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
        density_part = density_ft * gaussian_ft  # [B, f1, .., fd, in_channels]
        feature_part = feature_ft * gaussian_ft  # [B, f1, .., fd, in_channels]

        if self.density_feature_pairing == "blocked":
            return torch.cat([density_part, feature_part], dim=-1)

        # "interleaved": _to_group_format's contiguous split hands every group
        # its own density AND feature channels together, [D_g | F_g] — i.e. each
        # group's effective input is 2 * in_channels_per_group paired channels.
        # For groups == 1 the two layouts coincide.
        return einops.rearrange(
            [density_part, feature_part], "two ... (g c) -> ... (g two c)", g=self.groups
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
        # Rescale boundary coefficients (ξ_d = 0) by 0.5 via the precomputed
        # quadrature weights; out-of-place, so the caller's z_ft stays intact.
        z_ft = z_ft * self.ift_boundary_weights[..., None]

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

    def _apply(self, fn, *args, **kwargs):
        # Single choke point for .to()/.cuda()/.cpu()/.half()/etc. — all of them
        # route through _apply. Clear the Gaussian-FT cache so it rebuilds on
        # the new device/dtype instead of crashing the allclose comparison.
        self._cached_lengthscale = None
        self._cached_gaussian_ft = None
        out = super()._apply(fn, *args, **kwargs)

        # Module.to(dtype=<real dtype>) casts complex parameters to that real
        # dtype, silently discarding the imaginary half of the kernel weights.
        # Fail loudly rather than let the model continue with corrupted kernels.
        for name in ("weights", "U", "V"):
            param = getattr(self, name, None)
            if isinstance(param, torch.Tensor) and not param.is_complex():
                raise RuntimeError(
                    f"{type(self).__name__}.{name} lost its complex dtype "
                    f"(now {param.dtype}): a dtype cast such as "
                    f".to(torch.float32) discards the imaginary parts of the "
                    f"Fourier kernel weights. Use device-only moves instead "
                    f"(.to(device), .cuda(), .cpu())."
                )
        return out
