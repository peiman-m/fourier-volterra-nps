import torch

from ..core.decoders import TETNPDecoder
from ..core.encoders import TETNPEncoder
from ..likelihoods import BaseLikelihood
from .base import BaseNeuralProcess


class TETNP(BaseNeuralProcess):
    def __init__(
        self,
        encoder: TETNPEncoder,
        decoder: TETNPDecoder,
        likelihood: BaseLikelihood,
    ) -> None:
        super().__init__(encoder, decoder, likelihood)

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xq: torch.Tensor, **kwargs
    ) -> torch.distributions.Distribution:
        encoder_out = self.encoder(xc, yc, xq, **kwargs)
        decoder_out = self.decoder(encoder_out, xq)
        return self.likelihood(decoder_out)
