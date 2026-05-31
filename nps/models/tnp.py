import torch

from ..core.decoders import TNPDecoder
from ..core.encoders import TNPEncoder
from ..likelihoods import BaseLikelihood
from .base import BaseNeuralProcess


class TNP(BaseNeuralProcess):
    def __init__(
        self,
        encoder: TNPEncoder,
        decoder: TNPDecoder,
        likelihood: BaseLikelihood,
    ) -> None:
        super().__init__(encoder, decoder, likelihood)

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xq: torch.Tensor, **kwargs
    ) -> torch.distributions.Distribution:
        encoder_out = self.encoder(xc, yc, xq, **kwargs)
        decoder_out = self.decoder(encoder_out, xq)
        return self.likelihood(decoder_out)
