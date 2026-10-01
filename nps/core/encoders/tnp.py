import torch
import torch.nn as nn

from ...utils.helpers import preprocess_observations
from ..mlp import MLP
from ..transformers import (
    EfficientQueryTransformerEncoder,
    ISTransformerEncoder,
    PerceiverEncoder,
)
from .base import BaseEncoder


class TNPEncoder(BaseEncoder):
    def __init__(
        self,
        transformer_encoder: (
            EfficientQueryTransformerEncoder
            | ISTransformerEncoder
            | PerceiverEncoder
        ),
        xy_encoder: MLP,
        x_encoder: MLP | None = None,
        y_encoder: MLP | None = None,
    ) -> None:
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder or nn.Identity()
        self.y_encoder = y_encoder or nn.Identity()

    def _encode_inputs(
        self, xc: torch.Tensor, yc: torch.Tensor, xq: torch.Tensor, yq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nc, nq = xc.shape[1], xq.shape[1]

        # Encode x values
        x = torch.cat((xc, xq), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xq_encoded = x_encoded.split((nc, nq), dim=1)

        # Encode y values
        y = torch.cat((yc, yq), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yq_encoded = y_encoded.split((nc, nq), dim=1)

        # Concatenate and encode x,y pairs
        zc = self.xy_encoder(torch.cat((xc_encoded, yc_encoded), dim=-1))
        zq = self.xy_encoder(torch.cat((xq_encoded, yq_encoded), dim=-1))

        return zc, zq

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        # Preprocess observations (adding density channels, etc.)
        yc, yq = preprocess_observations(xq, yc)

        # Encode inputs
        zc, zq = self._encode_inputs(xc, yc, xq, yq)

        # Apply transformer encoder. These two-stream encoders enforce the
        # "queries attend to context, context attends only to itself" structure
        # architecturally, so no attention mask is needed.
        return self.transformer_encoder(zc, zq)
