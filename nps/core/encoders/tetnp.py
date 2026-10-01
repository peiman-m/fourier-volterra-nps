import torch

from ...utils.helpers import preprocess_observations
from ..mlp import MLP
from ..transformers import (
    TEEfficientQueryTransformerEncoder,
    TEISTransformerEncoder,
    TEPerceiverEncoder,
)
from .base import BaseEncoder


class TETNPEncoder(BaseEncoder):
    def __init__(
        self,
        transformer_encoder: (
            TEEfficientQueryTransformerEncoder
            | TEISTransformerEncoder
            | TEPerceiverEncoder
        ),
        y_encoder: MLP,
    ) -> None:
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.y_encoder = y_encoder

    def _encode_inputs(
        self, yc: torch.Tensor, yq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode y values separately
        zc = self.y_encoder(yc)
        zq = self.y_encoder(yq)

        return zc, zq

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        # Preprocess observations (adding density channels, etc.)
        yc, yq = preprocess_observations(xq, yc)

        # Encode observations
        zc, zq = self._encode_inputs(yc, yq)

        # Apply transformer encoder. These two-stream encoders enforce the
        # "queries attend to context, context attends only to itself" structure
        # architecturally, so no attention mask is needed.
        return self.transformer_encoder(zc, zq, xc, xq)
