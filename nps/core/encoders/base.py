from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Represents a neural process encoder base class"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass
