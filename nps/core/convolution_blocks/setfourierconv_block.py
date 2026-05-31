import copy

import torch
import torch.nn as nn

from ..convolutions import (
    SetFourierConv,
    SetFourierConvBase,
    SetFourierVolterraConv,
    SetFourierVolterraConvChunked,
)
from .base import BaseConvolutionBlock


class _SetFourierConvBlockBase(BaseConvolutionBlock):
    # Set by concrete subclasses to the SetFourierConv* variant to instantiate.
    _conv_cls: type[SetFourierConvBase] | None = None

    def __init__(
        self,
        spatial_dim: int,
        embed_channels: int,
        feedforward_channels: int | None = None,
        p_dropout: float = 0.0,
        activation: nn.Module | None = None,
        norm_first: bool = False,
        **kwargs
    ):
        super().__init__(
            spatial_dim=spatial_dim,
            in_channels=embed_channels,
            out_channels=embed_channels,
        )

        self.embed_channels = embed_channels

        feedforward_channels = (
            feedforward_channels
            if feedforward_channels is not None
            else embed_channels
        )
        self.feedforward_channels = feedforward_channels

        assert self._conv_cls is not None, "subclass must set _conv_cls"
        self.setfourierconv = self._conv_cls(
            spatial_dim=spatial_dim,
            in_channels=embed_channels,
            out_channels=embed_channels,
            **kwargs
        )

        activation = (
            nn.GELU()
            if activation is None
            else copy.deepcopy(activation)
        )

        self.ff_block = nn.Sequential(
            nn.Linear(embed_channels, feedforward_channels),
            activation,
            nn.Dropout(p_dropout),
            nn.Linear(feedforward_channels, embed_channels),
            nn.Dropout(p_dropout),
        )

        self.norm1 = nn.LayerNorm(embed_channels)
        self.norm2 = nn.LayerNorm(embed_channels)
        self.norm_first = norm_first

        self.dropout = nn.Dropout(p_dropout)

    def _conv_block(
        self,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return self.dropout(self.setfourierconv(zv=zv, xq=xq, xkv=xkv, **kwargs))

    def forward(
        self,
        zq: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.norm_first:
            zq = zq + self._conv_block(zv=self.norm1(zv), xq=xq, xkv=xkv, **kwargs)
            zq = zq + self.ff_block(self.norm2(zq))
        else:
            zq = self.norm1(zq + self._conv_block(zv=zv, xq=xq, xkv=xkv, **kwargs))
            zq = self.norm2(zq + self.ff_block(zq))
        return zq


class SetFourierConvBlock(_SetFourierConvBlockBase):
    _conv_cls = SetFourierConv


class SetFourierVolterraConvBlock(_SetFourierConvBlockBase):
    _conv_cls = SetFourierVolterraConv


class SetFourierVolterraConvChunkedBlock(_SetFourierConvBlockBase):
    _conv_cls = SetFourierVolterraConvChunked
