import math

import torch
from torch import nn

from .setfourierconvbase import SetFourierConvBase


class SetFourierConv(SetFourierConvBase):
    """
    Set-based Fourier convolution layer.

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

    def _build_kernel(self) -> None:
        """Initialize learnable Fourier domain kernel weights."""
        grid_shape = self.pos_half_freq_grid.shape[:-1]

        # Compute per-quadrant frequency grid shape
        # Only the first (spatial_dim - 1) dimensions are halved; last dimension is already positive-only
        freq_grid_shape = tuple(
            size // 2 + 1 for size in grid_shape[:-1]
        ) + (grid_shape[-1],)

        if self.low_rank:
            # low_rank=True guarantees rank was validated non-None in __init__.
            assert self.rank is not None
            # Low-rank factorization: W = U @ V
            # U: maps input to rank-dimensional space
            # V: maps rank-dimensional space to output
            self.U = nn.Parameter(
                torch.randn(
                    len(self.quad_slices),
                    self.groups,
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
                    self.groups,
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
                    len(self.quad_slices),
                    self.groups,
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

        # Initialize output tensor
        z_ft = torch.zeros(
            embedding_ft.shape[0],
            *self.pos_half_freq_grid.shape[:-1],  # [f1, .., fd, d]
            self.out_channels,
            dtype=torch.cfloat,
            device=embedding_ft.device,
        )

        # Apply convolution for each quadrant
        for q_idx, q_slc in enumerate(self.quad_slices):
            if self.low_rank:
                # Sequential contraction: avoids materialising W = U @ V.
                # Cost: O((C_eff + C_out) * rank * freq) vs O(C_eff * C_out * freq)
                projected = torch.einsum(
                    "b...gi, gir... -> b...gr", embedding_ft[q_slc], self.U[q_idx]
                )
                z_ft_quad = torch.einsum(
                    "b...gr, gro... -> b...go", projected, self.V[q_idx]
                )
            else:
                z_ft_quad = torch.einsum(
                    "b...gi, gio... -> b...go", embedding_ft[q_slc], self.weights[q_idx]
                )

            # Convert back from grouped format
            z_ft[q_slc] = self._from_group_format(z_ft_quad)

        # Apply inverse Fourier transform
        output = self._inverse_fourier(z_ft, xq, precomputed_ift_operands)

        # Apply output feature mixing if enabled
        if self.output_feature_mixing and self.output_mixing_layer is not None:
            output = self.output_mixing_layer(output)

        return output
