from typing import cast

import torch

from ...utils.helpers import get_clones
from ..convolution_blocks import SetFourierConvBlock, SetFourierVolterraConvBlock
from ..convolutions import SetFourierConvBase
from .base import BaseCNN


class SetFourierConvNet(BaseCNN):
    def __init__(
        self,
        block: SetFourierConvBlock | SetFourierVolterraConvBlock,
        num_blocks: int,
        share_params: bool = True,
    ):
        """
        Initialize the set Fourier convolution encoder.

        Args:
            block (SetFourierConvBlock | SetFourierVolterraConvBlock): Set Fourier Conv block to clone.
            num_blocks (int): Number of encoder blocks.
            share_params (bool): Whether to share parameters between context
                                 and query encoders.
        """
        super().__init__(
            spatial_dim=block.spatial_dim,
            in_channels=block.in_channels,
            out_channels=block.out_channels,
        )
        self.num_blocks = num_blocks

        self.query_encoder_blocks = get_clones(block, num_blocks)
        self.context_encoder_blocks = (
            self.query_encoder_blocks
            if share_params
            else get_clones(block, num_blocks)
        )

    def forward(
        self,
        zc: torch.Tensor,
        zq: torch.Tensor,
        xc: torch.Tensor,
        xq: torch.Tensor,
    ) -> torch.Tensor:
        # Precompute expensive operators once using first block's conv layer.
        # ModuleList indexing returns nn.Module; cast to the concrete conv type.
        first_conv = cast(
            SetFourierConvBase, self.context_encoder_blocks[0].setfourierconv
        )

        # Context operators (xq=xc, xkv=xc)
        context_translation_ft = first_conv.compute_translation_operands(xc)
        context_ift_operands = first_conv.compute_ift_operands(xc)

        # Query operators (xq=xq, xkv=xc)
        query_translation_ft = first_conv.compute_translation_operands(xc)
        query_ift_operands = first_conv.compute_ift_operands(xq)

        for context_encoder_block, query_encoder_block in zip(
            self.context_encoder_blocks, self.query_encoder_blocks
        ):
            zc = context_encoder_block(
                zq=zc,
                zv=zc,
                xq=xc,
                xkv=xc,
                precomputed_translation_ft=context_translation_ft,
                precomputed_ift_operands=context_ift_operands,
            )
            zq = query_encoder_block(
                zq=zq,
                zv=zc,
                xq=xq,
                xkv=xc,
                precomputed_translation_ft=query_translation_ft,
                precomputed_ift_operands=query_ift_operands,
            )

        return zq
