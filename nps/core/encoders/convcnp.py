from typing import Any

import einops
import torch
import torch.nn as nn

from ...utils.grids import construct_grid, flatten_grid
from ..convolutions import GridSetConv, SetConv
from ..mlp import MLP
from .base import BaseEncoder


class ConvCNPEncoder(BaseEncoder):

    def __init__(
        self,
        convnet: nn.Module,
        grid_encoder: SetConv,
        grid_decoder: SetConv,
        z_encoder: MLP,
        grid_resolution: float | tuple[float, ...],
        use_pos_encoding: bool = False,
        x_encoder: MLP | None = None,
        grid_margin: (
            float
            | tuple[float, ...]  # dims * margin (same left/right)
            | tuple[float, float]  # [left_margin, right_margin] for all dims
            | tuple[tuple[float, float], ...]  # dims * [left_margin, right_margin]
        ) = 0.0,
        grid_size_multiple_of: int | tuple[int, ...] = 1,
        grid_span_adjust_mode: str | tuple[str, ...] = "both",
        grid_size_divisibility_adjust_mode: str | tuple[str, ...] = "increase",
    ) -> None:
        super().__init__()
        self.convnet = convnet
        self.grid_encoder = grid_encoder
        self.grid_decoder = grid_decoder
        self.z_encoder = z_encoder
        self.use_pos_encoding = use_pos_encoding
        self.x_encoder = x_encoder or nn.Identity()

        # Store grid construction parameters
        self.grid_resolution = grid_resolution
        self.grid_margin = grid_margin
        self.grid_size_multiple_of = grid_size_multiple_of
        self.grid_span_adjust_mode = grid_span_adjust_mode
        self.grid_size_divisibility_adjust_mode = grid_size_divisibility_adjust_mode

    def _get_grid_params(
        self,
        xc: torch.Tensor,
        xq: torch.Tensor,
    ) -> dict[str, Any]:
        # Combine context and query x to get global span
        x_all = torch.cat((xc, xq), dim=1)  # [B, Nc+Nq, d]
        B, _, d = x_all.shape
        x_flat = einops.rearrange(x_all, "... d -> (...) d")
        lo, _ = x_flat.min(dim=0)
        hi, _ = x_flat.max(dim=0)
        span = torch.stack((lo, hi), dim=-1)  # [d, 2]
        return {
            "span": span,
            "dim": d,
            "resolution": self.grid_resolution,
            "batch_size": B,
            "margin": self.grid_margin,
            "multiple_of": self.grid_size_multiple_of,
            "span_adjust_mode": self.grid_span_adjust_mode,
            "size_divisibility_adjust_mode": self.grid_size_divisibility_adjust_mode,
            "device": xc.device,
        }

    def forward(
        self,
        xc: torch.Tensor,  # [B, K, d]
        yc: torch.Tensor,  # [B, K, C]
        xq: torch.Tensor,  # [B, Q, d]
    ) -> torch.Tensor:  # -> [B, Q, C_out]
        # build a regular grid covering both xc and xq
        x_grid = construct_grid(**self._get_grid_params(xc, xq))

        # flatten it so we can do set‐conv
        x_grid_flat, unflatten_fn = flatten_grid(x_grid)  # [B, M, d]

        # project context (xc,yc) onto grid
        z_flat = self.grid_encoder(xkv=xc, xq=x_grid_flat, zv=yc)
        z_grid = unflatten_fn(z_flat)  # [B, *grid_shape, C_z]

        # point‐wise feature encoder
        z_grid = self.z_encoder(z_grid)
        z_grid = einops.rearrange(z_grid, "b ... c -> b c ...")

        if self.use_pos_encoding:
            x_pe = self.x_encoder(x_grid)  # [B, *grid_shape, C_pe] or identity
            x_pe = einops.rearrange(x_pe, "b ... c -> b c ...")
            z = (z_grid, x_pe)
        else:
            z = (z_grid,)

        z_grid = self.convnet(*z)

        # restore grid ordering: (B, ..., C)
        z_grid = einops.rearrange(z_grid, "b c ... -> b ... c")

        # flatten and decode back to your query points xq
        z_grid_flat, _ = flatten_grid(z_grid)  # [B, M, C_out]
        return self.grid_decoder(xkv=x_grid_flat, xq=xq, zv=z_grid_flat)


class GridConvCNPEncoder(BaseEncoder):

    def __init__(
        self,
        convnet: nn.Module,
        grid_encoder: GridSetConv,
        z_encoder: MLP,
    ) -> None:
        super().__init__()
        self.convnet = convnet
        self.grid_encoder = grid_encoder
        self.z_encoder = z_encoder

    def forward(
        self,
        y_mc: torch.Tensor,  # equivalent to xc
        y: torch.Tensor,  # equivalent to yc
        y_mq: torch.Tensor,  # equivalent to xq
    ) -> torch.Tensor:
        # encode context mask+values into a grid
        z_grid = self.grid_encoder(xkv_mask=y_mc, zv=y)

        # point‐wise encoding followed by convolution
        z_grid = self.z_encoder(z_grid)
        z_grid = self.convnet(z_grid)

        return self._extract_query(z_grid, y_mq)

    @staticmethod
    def _extract_query(
        z_grid: torch.Tensor,  # [B, *grid_shape, C]
        mask: torch.Tensor,  # [B, *grid_shape]
    ) -> torch.Tensor:
        B = z_grid.shape[0]
        z_flat = einops.rearrange(z_grid, "b c ... -> b (...) c")
        m_flat = einops.rearrange(mask[:, 0], "b ... -> b (...)")
        # for each batch, pick the masked rows
        zq = [z_flat[b][m_flat[b].bool()] for b in range(B)]
        return torch.stack(zq, dim=0)  # [B, N_b, C]
