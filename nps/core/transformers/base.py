from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTransformerEncoder(nn.Module, ABC):
    """
    Abstract base class for transformer encoder.
    """

    def __init__(
        self,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass
