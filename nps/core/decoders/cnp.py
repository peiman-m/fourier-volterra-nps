import torch

from ..mlp import MLP
from .base import BaseDecoder


class CNPDecoder(BaseDecoder):
    def __init__(
        self,
        z_decoder: MLP,
    ) -> None:
        super().__init__()
        self.z_decoder = z_decoder

    def forward(self, z: torch.Tensor, xq: torch.Tensor | None = None) -> torch.Tensor:
        # Process query points if provided
        zq = z if xq is None else z[..., -xq.shape[-2] :, :]
        zq = self.z_decoder(zq)
        return zq
