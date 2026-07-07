import torch

from ...utils.helpers import preprocess_observations
from ..cnns.sfcnn import SetFourierConvNet
from ..mlp import MLP
from .base import BaseEncoder


class SetFourierConvCNPEncoder(BaseEncoder):
    def __init__(
        self,
        set_encoder: SetFourierConvNet,
        y_encoder: MLP,
    ) -> None:
        super().__init__()

        self.set_encoder = set_encoder
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

        return self.set_encoder(zc=zc, zq=zq, xc=xc, xq=xq)
