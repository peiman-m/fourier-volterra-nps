import math
import warnings

import einops
import torch
from torch import nn

from .setfourierconvbase import SetFourierConvBase


class SetFourierVolterraConvChunked(SetFourierConvBase):
    """
    Set-based Fourier Volterra convolution with adaptive chunked forward pass.

    Identical interface to SetFourierVolterraConv. Differences:

    1. Weight layout — pair-interleaved:
         [linear | pair0_f1 | pair0_f2 | pair1_f1 | pair1_f2 | ...]
       Both classes use this layout. Checkpoints are fully interchangeable.

    2. Memory — the nonlinear R pairs are processed in chunks of `_chunk_pairs`
       rather than all at once. Peak allocation scales with chunk size instead
       of R, enabling larger models or batch sizes on memory-constrained GPUs.

    3. OOM recovery — if a forward pass raises OutOfMemoryError in the
       nonlinear chunk loop, `_chunk_pairs` is permanently halved and the pass
       is retried automatically. The linear term is never recomputed on retry.

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
        volterra_rank (int): Approximation rank of the 2nd order Volterra filter.
        input_feature_mixing (bool): Whether to apply linear mixing to input features.
        output_feature_mixing (bool): Whether to apply linear mixing to output features.
    """

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
        volterra_rank: int = 4,
        input_feature_mixing: bool = False,
        output_feature_mixing: bool = False,
    ) -> None:
        if volterra_rank is None or not (1 <= volterra_rank):
            raise ValueError(
                f"volterra_rank must be specified and satisfy 1 <= volterra_rank, "
                f"but got volterra_rank={volterra_rank}"
            )
        # Must be set before super().__init__() since _build_kernel uses it
        self.volterra_rank = volterra_rank

        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            max_freq=max_freq,
            freq_resolution=freq_resolution,
            groups=groups,
            init_lengthscale=init_lengthscale,
            learnable_lengthscale=learnable_lengthscale,
            legnthscale_cache_tolerance=legnthscale_cache_tolerance,
            low_rank=low_rank,
            rank=rank,
            input_feature_mixing=input_feature_mixing,
            output_feature_mixing=output_feature_mixing,
        )

        # Linear layer for aggregating the product terms of the low-rank approximation
        self.low_ranks_mixer = nn.Linear(volterra_rank, 1)

        # Start fully parallel (all R pairs at once); halved permanently on OOM
        self._chunk_pairs: int = self.volterra_rank

    def to(self, *args, **kwargs):
        super_out = super().to(*args, **kwargs)  # clears lengthscale cache
        self._chunk_pairs = self.volterra_rank   # reset: new device, fresh memory budget
        return super_out

    def _build_kernel(self) -> None:
        """Initialize learnable Fourier domain kernel weights.

        Weight layout along the groups axis is pair-interleaved:
            block 0          : linear term
            blocks 1, 2      : pair 0  (factor 1, factor 2)
            blocks 3, 4      : pair 1  (factor 1, factor 2)
            ...
            blocks 2R-1, 2R  : pair R-1
        Each block is G groups wide.
        """
        grid_shape = self.pos_half_freq_grid.shape[:-1]

        # Compute per-quadrant frequency grid shape
        # Only the first (spatial_dim - 1) dimensions are halved; last dimension is already positive-only
        freq_grid_shape = tuple(
            size // 2 + 1 for size in grid_shape[:-1]
        ) + (grid_shape[-1],)

        # Groups dimension: 1 linear block + 2*volterra_rank nonlinear blocks, each G wide
        kernel_groups = self.groups * (2 * self.volterra_rank + 1)

        if self.low_rank:
            # low_rank=True guarantees rank was validated non-None in __init__.
            assert self.rank is not None
            self.U = nn.Parameter(
                torch.randn(
                    len(self.quad_slices),
                    kernel_groups,
                    self.effective_in_channels_per_group,
                    self.rank,
                    *freq_grid_shape,
                    dtype=torch.cfloat,
                )
                * (1.0 / math.sqrt(self.in_channels * self.rank))
            )
            self.V = nn.Parameter(
                torch.randn(
                    len(self.quad_slices),
                    kernel_groups,
                    self.rank,
                    self.out_channels_per_group,
                    *freq_grid_shape,
                    dtype=torch.cfloat,
                )
                * (1.0 / math.sqrt(self.out_channels * self.rank))
            )
        else:
            self.weights = nn.Parameter(
                torch.randn(
                    len(self.quad_slices),
                    kernel_groups,
                    self.effective_in_channels_per_group,
                    self.out_channels_per_group,
                    *freq_grid_shape,
                    dtype=torch.cfloat,
                )
                * (1.0 / math.sqrt(self.in_channels * self.out_channels))
            )

    def _forward_linear_term(
        self,
        embedding_ft_grouped: torch.Tensor,        # [B, *freq, G, C_eff]
        xq: torch.Tensor,
        precomputed_ift_operands: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute z_order1 from weight block 0. Returns [B, N_q, out_channels]."""
        B = embedding_ft_grouped.shape[0]
        device = embedding_ft_grouped.device
        w_start, w_end = 0, self.groups

        z_ft_linear = torch.zeros(
            B,
            *self.pos_half_freq_grid.shape[:-1],
            self.out_channels,
            dtype=torch.cfloat,
            device=device,
        )

        for q_idx, q_slc in enumerate(self.quad_slices):
            if self.low_rank:
                U_q = self.U[q_idx, w_start:w_end]
                V_q = self.V[q_idx, w_start:w_end]
                projected = torch.einsum("b...gi, gir... -> b...gr", embedding_ft_grouped[q_slc], U_q)
                z_ft_quad = torch.einsum("b...gr, gro... -> b...go", projected, V_q)
            else:
                z_ft_quad = torch.einsum(
                    "b...gi, gio... -> b...go",
                    embedding_ft_grouped[q_slc],
                    self.weights[q_idx, w_start:w_end],
                )
            z_ft_linear[q_slc] = self._from_group_format(z_ft_quad)

        z_order1 = self._inverse_fourier(z_ft_linear, xq, precomputed_ift_operands)
        del z_ft_linear
        return z_order1

    def _forward_nonlinear_chunked(
        self,
        chunk_pairs: int,
        embedding_ft_grouped: torch.Tensor,        # [B, *freq, G, C_eff]
        xq: torch.Tensor,
        precomputed_ift_operands: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute z_order2 over all R pairs in chunks. Returns [B, N_q, out_channels]."""
        B = embedding_ft_grouped.shape[0]
        N_q = xq.shape[1]
        device = embedding_ft_grouped.device

        # Extract mixer weights once; gradients flow back through the view
        mixer_w = self.low_ranks_mixer.weight[0]  # [R]

        z_order2_accum = torch.zeros(B, N_q, self.out_channels, dtype=torch.float32, device=device)

        for start_pair in range(0, self.volterra_rank, chunk_pairs):
            end_pair = min(start_pair + chunk_pairs, self.volterra_rank)
            k_pairs  = end_pair - start_pair
            k_terms  = 2 * k_pairs  # always even

            # Contiguous weight slice for pairs [start_pair, end_pair)
            # Pair-interleaved layout: pair r occupies blocks 2r+1 and 2r+2
            w_start = (2 * start_pair + 1) * self.groups
            w_end   = (2 * end_pair   + 1) * self.groups

            if not self.low_rank:
                weights_chunk = self.weights[:, w_start:w_end, ...]

            # Expand embedding for k_terms terms: [B, *freq, k_terms*G, C_eff]
            emb_chunk = einops.repeat(
                embedding_ft_grouped, "... g c -> ... (r g) c", r=k_terms
            )

            z_ft_chunk = torch.zeros(
                B,
                *self.pos_half_freq_grid.shape[:-1],
                self.out_channels * k_terms,
                dtype=torch.cfloat,
                device=device,
            )

            for q_idx, q_slc in enumerate(self.quad_slices):
                if self.low_rank:
                    U_q = self.U[q_idx, w_start:w_end]
                    V_q = self.V[q_idx, w_start:w_end]
                    projected = torch.einsum("b...gi, gir... -> b...gr", emb_chunk[q_slc], U_q)
                    z_ft_quad = torch.einsum("b...gr, gro... -> b...go", projected, V_q)
                else:
                    z_ft_quad = torch.einsum(
                        "b...gi, gio... -> b...go", emb_chunk[q_slc], weights_chunk[q_idx]
                    )
                z_ft_chunk[q_slc] = self._from_group_format(z_ft_quad)

            z_chunk = self._inverse_fourier(z_ft_chunk, xq, precomputed_ift_operands)
            # [B, N_q, k_terms * out_channels]

            z_chunk = einops.rearrange(z_chunk, "... (r c) -> ... r c", c=self.out_channels)
            # [B, N_q, k_terms, out_channels]

            z_pairs = einops.rearrange(z_chunk, "... (p two) c -> ... p two c", two=2)
            # [B, N_q, k_pairs, 2, out_channels]

            z_products = z_pairs[..., 0, :] * z_pairs[..., 1, :]
            # [B, N_q, k_pairs, out_channels]

            for i in range(k_pairs):
                z_order2_accum = z_order2_accum + mixer_w[start_pair + i] * z_products[:, :, i, :]

            del emb_chunk, z_ft_chunk, z_chunk, z_pairs, z_products
            # Do NOT call torch.cuda.empty_cache() here — expensive in the non-OOM case.
            # The retry handler in forward() calls it once after catching OutOfMemoryError.

        return z_order2_accum + self.low_ranks_mixer.bias[0]

    def forward(
        self,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        precomputed_translation_ft: torch.Tensor | None = None,
        precomputed_ift_operands: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform set convolution from (xkv, zv) to xq.
        """
        if self.input_feature_mixing and self.input_mixing_layer is not None:
            zv = self.input_mixing_layer(zv)

        # Cheap steps that must not be repeated — outside the retry loop
        embedding_ft = self._forward_fourier(zv, xkv, precomputed_translation_ft)
        embedding_ft = self._to_group_format(embedding_ft)  # [B, *freq, G, C_eff]

        # Linear term: constant cost, independent of chunk_pairs — outside the retry loop
        z_order1 = self._forward_linear_term(embedding_ft, xq, precomputed_ift_operands)

        # Nonlinear pair loop: OOM-aware retry
        while True:
            try:
                z_order2 = self._forward_nonlinear_chunked(
                    self._chunk_pairs,
                    embedding_ft,
                    xq,
                    precomputed_ift_operands,
                )
                break

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self._chunk_pairs == 1:
                    raise RuntimeError(
                        f"SetFourierVolterraConvChunked OOM with chunk_pairs=1. "
                        f"Cannot reduce further (volterra_rank={self.volterra_rank}). "
                        f"Reduce batch size, volterra_rank, or frequency grid resolution."
                    ) from None
                self._chunk_pairs = max(1, self._chunk_pairs // 2)
                warnings.warn(
                    f"SetFourierVolterraConvChunked: OOM — reducing chunk_pairs to "
                    f"{self._chunk_pairs} (volterra_rank={self.volterra_rank}). "
                    f"To avoid this, reduce batch size, volterra_rank, or freq_resolution."
                )
                # loop continues with the reduced chunk size

        z = z_order1 + z_order2

        if self.output_feature_mixing and self.output_mixing_layer is not None:
            return self.output_mixing_layer(z)

        return z
