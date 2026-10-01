import torch
import torch.nn as nn

from ..attention_layers import MultiHeadAttentionLayer
from ..mlp import MLP
from ..transformers import TransformerEncoder
from .base import BaseEncoder


class ACNPEncoder(BaseEncoder):
    def __init__(
        self,
        transformer_encoder: TransformerEncoder,
        mha_layer: MultiHeadAttentionLayer,
        xy_encoder: TransformerEncoder,
        x_encoder: MLP | None = None,
        y_encoder: MLP | None = None,
    ) -> None:
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.mha_layer = mha_layer
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder or nn.Identity()
        self.y_encoder = y_encoder or nn.Identity()

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((xc, xq), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xq_encoded = x_encoded.split((xc.shape[1], xq.shape[1]), dim=1)

        yc_encoded = self.y_encoder(yc)

        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zc = self.xy_encoder(zc)

        # Self-attention layers on context tokens.
        zc = self.transformer_encoder(zc)

        # Cross-attention layer with input locations as query/keys.
        return self.mha_layer(xq_encoded, xc_encoded, zc)
