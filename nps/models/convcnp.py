import torch

from ..core.decoders import ConvCNPDecoder
from ..core.encoders import ConvCNPEncoder, GridConvCNPEncoder
from ..likelihoods import BaseLikelihood
from .base import BaseNeuralProcess


class ConvCNP(BaseNeuralProcess):
    def __init__(
        self,
        encoder: ConvCNPEncoder,
        decoder: ConvCNPDecoder,
        likelihood: BaseLikelihood,
    ) -> None:
        super().__init__(encoder, decoder, likelihood)

    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xq: torch.Tensor,
    ) -> torch.distributions.Distribution:
        encoder_out = self.encoder(xc, yc, xq)
        decoder_out = self.decoder(encoder_out)
        return self.likelihood(decoder_out)


class GridConvCNP(BaseNeuralProcess):
    def __init__(
        self,
        encoder: GridConvCNPEncoder,
        decoder: ConvCNPDecoder,
        likelihood: BaseLikelihood,
    ) -> None:
        super().__init__(encoder, decoder, likelihood)

    def forward(
        self,
        y_mc: torch.Tensor,
        y: torch.Tensor,
        y_mq: torch.Tensor,
    ) -> torch.distributions.Distribution:
        encoder_out = self.encoder(y_mc, y, y_mq)
        decoder_out = self.decoder(encoder_out)
        return self.likelihood(decoder_out)
