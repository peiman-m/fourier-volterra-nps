import torch

from ...utils.helpers import preprocess_observations
from ..mlp import MLP
from ..transformers import (
    TEEfficientQueryTransformerEncoder,
    TEISTransformerEncoder,
    TEPerceiverEncoder,
    TETransformerEncoder,
)
from .base import BaseEncoder


class TETNPEncoder(BaseEncoder):
    def __init__(
        self,
        transformer_encoder: (
            TETransformerEncoder
            | TEEfficientQueryTransformerEncoder
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

    def _create_attention_mask(
        self, batch_size: int, nc: int, nq: int, device: torch.device
    ) -> torch.Tensor:
        # Create a mask where only context points can be attended to
        total_points = nc + nq
        mask = torch.full(
            (batch_size, total_points, total_points), float("-inf"), device=device
        )
        mask[:, :, :nc] = 0.0

        return mask

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        # Preprocess observations (adding density channels, etc.)
        yc, yq = preprocess_observations(xq, yc)

        # Encode observations
        zc, zq = self._encode_inputs(yc, yq)

        # Create attention mask
        batch_size, nc, nq = *xc.shape[:2], xq.shape[1]
        mask = self._create_attention_mask(batch_size, nc, nq, xc.device)

        # Apply transformer encoder
        return self.transformer_encoder(zc, zq, xc, xq, mask)
