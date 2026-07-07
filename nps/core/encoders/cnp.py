import einops
import torch
import torch.nn as nn

from ..deepset import DeepSet
from ..mlp import MLP
from .base import BaseEncoder


class CNPEncoder(BaseEncoder):
    def __init__(
        self,
        deepset: DeepSet,
        x_encoder: MLP | None = None,
        y_encoder: MLP | None = None,
    ) -> None:
        super().__init__()
        self.deepset = deepset
        self.x_encoder = x_encoder or nn.Identity()
        self.y_encoder = y_encoder or nn.Identity()

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xq: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((xc, xq), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xq_encoded = x_encoded.split((xc.shape[1], xq.shape[1]), dim=1)

        yc_encoded = self.y_encoder(yc)

        zc = self.deepset(xc_encoded, yc_encoded, **kwargs)

        # Use same context representation for every query point.
        zc = einops.repeat(zc, "b d -> b nq d", nq=xq.shape[-2])

        # Concatenate xq to zc.
        return torch.cat((zc, xq_encoded), dim=-1)
