from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..core.decoders import BaseDecoder
from ..core.encoders import BaseEncoder
from ..likelihoods import BaseLikelihood


class BaseNeuralProcess(nn.Module, ABC):
    """Represents a neural process base class"""

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        likelihood: BaseLikelihood,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.distributions.Distribution:
        pass
