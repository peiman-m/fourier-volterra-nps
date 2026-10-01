from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseDecoder(nn.Module, ABC):
    """Represents a neural process decoder base class"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass
