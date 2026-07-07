import itertools
from collections.abc import Sequence
from functools import partial
from typing import Any

import einops
import torch
import torch.nn as nn

from .base import BaseConvolution


class SpectralConv(BaseConvolution):
    """
    Spectral convolution in N dimensions using Fourier transforms.

    Args:
        spatial_dim (int): Number of spatial dimensions (1, 2, or 3).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (int | Sequence[int]): Number of Fourier modes to keep per dimension.
        groups (int | None): Number of channel groups. Defaults to 1.
        low_rank (bool): Whether to use low-rank factorization of complex kernel weights.
        rank (int | None): Rank for low-rank factorization.
        norm (str | None): Normalization for FFT ('backward', 'ortho', 'forward').
            Defaults to None (PyTorch default).

    Note:
        Ensure that the input spatial size s_d at dimension d.
        TODO: Add checks for this
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        modes: int | Sequence[int],
        groups: int | None = None,
        low_rank: bool = False,
        rank: int | None = None,
        fft_norm: str | None = "backward",
    ):
        super().__init__(
            spatial_dim=spatial_dim, in_channels=in_channels, out_channels=out_channels
        )
        self.groups = groups or 1
        self.low_rank = low_rank
        self.rank = rank
        self.fft_norm = fft_norm

        # Validate inputs
        if in_channels % self.groups != 0 or out_channels % self.groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) and out_channels ({out_channels}) "
                f"must be divisible by groups ({self.groups})"
            )

        # Store channels per group for easier reference
        self.in_channels_per_group = in_channels // self.groups
        self.out_channels_per_group = out_channels // self.groups

        # Normalize & validate modes
        self.modes = self._normalize_modes(modes)
        if any(m < 1 for m in self.modes):
            raise ValueError("All `modes` values must be ≥ 1")

        # Validate low-rank parameters
        if low_rank:
            if rank is None or not (
                1
                <= rank
                <= min(self.in_channels_per_group, self.out_channels_per_group)
            ):
                raise ValueError(
                    f"For low_rank=True, rank must satisfy"
                    f"1 <= rank <= min(in_channels_per_group, out_channels_per_group), "
                    f"got rank={rank}, "
                    f"in_channels_per_group={self.in_channels_per_group}, "
                    f"out_channels_per_group={self.out_channels_per_group}"
                )

        # FFT / IFFT functions with optional norm
        fft_opts: dict[str, Any] = {"dim": tuple(range(-spatial_dim, 0))}
        if fft_norm is not None:
            fft_opts["norm"] = fft_norm
        self.fft_fn = partial(torch.fft.rfftn, **fft_opts)
        self.ifft_fn = partial(torch.fft.irfftn, **fft_opts)

        # Precompute quadrant index slices
        self._construct_quad_slices()

        # Initialize spectral weights
        self._build_kernel()

    def _normalize_modes(self, modes: int | Sequence[int]) -> tuple[int, ...]:
        if isinstance(modes, int):
            return (modes,) * self.spatial_dim
        elif isinstance(modes, Sequence):
            if len(modes) != self.spatial_dim:
                raise ValueError(f"Expected {self.spatial_dim} modes, got {len(modes)}")
            return tuple(modes)
        else:
            raise TypeError("Invalid type for `modes`")

    def _normalize_size(
        self, size: int | Sequence[int] | None, default: tuple[int, ...]
    ) -> tuple[int, ...]:
        if size is None:
            return tuple(default)
        elif isinstance(size, int):
            return (size,) * self.spatial_dim
        elif isinstance(size, Sequence):
            if len(size) != self.spatial_dim:
                raise ValueError(
                    f"Expected size sequence of length {self.spatial_dim}, "
                    f"got {len(size)}"
                )
            return tuple(size)
        else:
            raise TypeError(f"Invalid type for size: {type(size)}")

    def _build_kernel(self):
        """Initialize weights with uniform distribution"""
        num_quadrants = 2 ** (self.spatial_dim - 1)

        if self.low_rank:
            # low_rank=True guarantees rank was validated non-None in __init__.
            assert self.rank is not None
            # Low-rank factorization: W = U @ V
            U_init_scale = 1.0 / (self.groups * self.in_channels_per_group * self.rank)
            V_init_scale = 1.0 / (self.groups * self.rank * self.out_channels_per_group)

            U_size = (
                num_quadrants,
                self.groups,
                self.in_channels_per_group,
                self.rank,
                *self.modes,
            )
            V_size = (
                num_quadrants,
                self.groups,
                self.rank,
                self.out_channels_per_group,
                *self.modes,
            )

            self.U = nn.Parameter(
                U_init_scale * torch.complex(torch.rand(*U_size), torch.rand(*U_size))
            )
            self.V = nn.Parameter(
                V_init_scale * torch.complex(torch.rand(*V_size), torch.rand(*V_size))
            )
        else:
            # Standard full-rank weights
            W_scale = 1.0 / (
                self.groups * self.in_channels_per_group * self.out_channels_per_group
            )

            W_size = (
                num_quadrants,
                self.groups,
                self.in_channels_per_group,
                self.out_channels_per_group,
                *self.modes,
            )

            self.weights = nn.Parameter(
                W_scale * torch.complex(torch.rand(W_size), torch.rand(W_size))
            )

    def _to_group_format(self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(tensor, "b (g c) ... -> b g c ...", g=self.groups)

    def _from_group_format(self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(tensor, "b g c ... -> b (g c) ...")

    def _construct_quad_slices(self) -> None:
        """Build slice objects for different frequency quadrants."""
        slices: list[tuple] = []
        for q_sign in itertools.product([0, 1], repeat=self.spatial_dim - 1):
            q_slc = []
            for d, sign in enumerate(q_sign):
                m = self.modes[d]
                if sign == 0:
                    q_slc.append(slice(None, m))
                elif sign == 1:
                    q_slc.append(slice(-m, None))
                else:
                    raise ValueError(f"Invalid sign {sign} in {q_sign}.")

            q_slc.append(
                slice(None, self.modes[-1])
            )  # only positive frequencies for last dim
            q_slc = (...,) + tuple(q_slc)  # add ellipsis for groups and channels
            slices.append(q_slc)

        self.quad_slices = slices

    def forward(
        self,
        x: torch.Tensor,
        *,
        in_size: int | tuple[int, ...] | None = None,
        out_size: int | tuple[int, ...] | None = None,
    ) -> torch.Tensor:

        in_size = self._normalize_size(in_size, x.shape[-self.spatial_dim :])
        out_size = self._normalize_size(out_size, x.shape[-self.spatial_dim :])

        # Check min size
        for d, m in enumerate(self.modes):
            if in_size[d] // 2 + 1 < m:
                raise ValueError(
                    f"For {in_size[d]} samples at dim={d}, there will be at most "
                    f"{in_size[d] // 2 + 1} modes, but need {m} modes. "
                )

        # Compute Fourier coefficients with n=in_size
        x_ft = self.fft_fn(x, s=in_size)
        x_ft = self._to_group_format(x_ft)

        z_ft = torch.zeros(
            x_ft.shape[0],
            self.out_channels,
            *x_ft.shape[-self.spatial_dim :],
            dtype=x_ft.dtype,
            device=x_ft.device,
        )

        if self.low_rank:
            weights = torch.einsum("qgir..., qgro... -> qgio...", self.U, self.V)
        else:
            weights = self.weights

        for q_idx, q_s in enumerate(self.quad_slices):
            z_ft_q = torch.einsum("bgi..., gio... -> bgo...", x_ft[q_s], weights[q_idx])
            z_ft[q_s] = self._from_group_format(z_ft_q)

        # Return to physical space
        return self.ifft_fn(z_ft, s=out_size).real
