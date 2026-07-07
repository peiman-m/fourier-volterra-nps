import math

import einops
import torch
from torch import nn

from .setfourierconvbase import SetFourierConvBase


class SetFourierVolterraConv(SetFourierConvBase):
    """
    Set-based Fourier Volterra convolution layer.

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
        volterra_rank (int): Approximation rank of the 2nd order Volterra filter.
        input_feature_mixing (bool): Whether to apply linear mixing to input features.
        output_feature_mixing (bool): Whether to apply linear mixing to output features.
        freq_chunks (int | None): Number of disjoint chunks along the first
            frequency axis for the spectral product (memory/compute schedule
            only; results are identical for any value).
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
        lengthscale_cache_tolerance: float = 1e-5,
        low_rank: bool = False,
        rank: int | None = None,
        volterra_rank: int = 4,
        input_feature_mixing: bool = False,
        output_feature_mixing: bool = False,
        freq_chunks: int | None = None,
        density_feature_pairing: str = "blocked",
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
            lengthscale_cache_tolerance=lengthscale_cache_tolerance,
            low_rank=low_rank,
            rank=rank,
            input_feature_mixing=input_feature_mixing,
            output_feature_mixing=output_feature_mixing,
            freq_chunks=freq_chunks,
            density_feature_pairing=density_feature_pairing,
        )

        # Linear layer for aggregating the product terms of the low-rank approximation
        self.low_ranks_mixer = nn.Linear(volterra_rank, 1)

    def _build_kernel(self) -> None:
        """Initialize learnable Fourier domain kernel weights on the full grid."""
        freq_grid_shape = self.pos_half_freq_grid.shape[:-1]

        # Groups dimension: 1 linear block + 2*volterra_rank nonlinear blocks, each G wide.
        # Pair-interleaved layout: [linear | pair0_f1 | pair0_f2 | pair1_f1 | pair1_f2 | ...]
        kernel_groups = self.groups * (2 * self.volterra_rank + 1)

        if self.low_rank:
            # low_rank=True guarantees rank was validated non-None in __init__.
            assert self.rank is not None
            # Low-rank factorization: W = U @ V
            # U: maps input to rank-dimensional space
            # V: maps rank-dimensional space to output
            self.U = nn.Parameter(
                torch.randn(
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
                    kernel_groups,
                    self.rank,
                    self.out_channels_per_group,
                    *freq_grid_shape,
                    dtype=torch.cfloat,
                )
                * (1.0 / math.sqrt(self.out_channels * self.rank))
            )
        else:
            # Full-rank kernel weights
            self.weights = nn.Parameter(
                torch.randn(
                    kernel_groups,
                    self.effective_in_channels_per_group,
                    self.out_channels_per_group,
                    *freq_grid_shape,
                    dtype=torch.cfloat,
                )
                * (1.0 / math.sqrt(self.in_channels * self.out_channels))
            )

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
        # Apply input feature mixing if enabled
        if self.input_feature_mixing and self.input_mixing_layer is not None:
            zv = self.input_mixing_layer(zv)

        # Compute Fourier transform of functional embedding
        # [B, f1, .., fd, 2*in_channels]
        embedding_ft = self._forward_fourier(zv, xkv, precomputed_translation_ft)

        # Convert to grouped format for grouped convolution
        # [B, f1, .., fd, G, C_eff_per_group]
        embedding_ft = self._to_group_format(embedding_ft)

        # Apply the Fourier-domain kernel. The (2R+1) Volterra blocks are
        # broadcast inside the spectral product, so the embedding is never
        # materialized (2R+1)-fold.
        # [B, f1, .., fd, (2R+1) * out_channels]  (1 for the linear term)
        z_ft = self._spectral_product(embedding_ft)

        z = self._inverse_fourier(z_ft, xq, precomputed_ift_operands)

        z = einops.rearrange(z, '... (r c) -> ... r c', c=self.out_channels)

        # Extract 1st and 2nd order terms
        # Pair-interleaved layout: z[..., 1:, :] has shape [B, N_q, 2R, out_channels]
        # Consecutive pairs: (t=1,t=2) → pair 0, (t=3,t=4) → pair 1, ...
        z_order1 = z[..., 0, :]
        z_nonlinear = einops.rearrange(z[..., 1:, :], '... (p two) c -> ... p two c', two=2)
        z_order2 = z_nonlinear[..., 0, :] * z_nonlinear[..., 1, :]
        # [B, N_q, R, out_channels]
        z_order2 = einops.rearrange(z_order2, '... r c -> ... c r')

        # Aggregate the product terms
        z_order2 = self.low_ranks_mixer(z_order2).squeeze(-1)

        # Apply output feature mixing if enabled
        if self.output_feature_mixing and self.output_mixing_layer is not None:
            return self.output_mixing_layer(z_order1 + z_order2)

        return z_order1 + z_order2
