from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseMultiHeadAttention(nn.Module, ABC):
    """
    Abstract base class for multi-head attention modules.
    """

    def __init__(
        self,
        *,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        num_heads: int,
        head_dim: int,
        p_dropout: float = 0.0,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.p_dropout = p_dropout
        self.scale = head_dim**-0.5 if scale is None else scale

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass

    # TODO: implement this method
    # @abstractmethod
    # @torch.no_grad()
    # def get_attn_weights(self, *args, **kwargs) -> torch.Tensor:
    #     pass
