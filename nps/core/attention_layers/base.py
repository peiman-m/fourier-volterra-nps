import copy
from abc import ABC, abstractmethod

import torch.nn as nn

from ..attentions import BaseMultiHeadAttention


class BaseMultiHeadAttentionLayer(nn.Module, ABC):
    """
    Base class for multi-head attention layers.

    This abstract class provides the foundation for different
    types of multi-head attention mechanisms including standard
    attention and translation equivariant attention.

    Args:
        attention: The attention mechanism to use
        embed_dim: Dimension of the embedding vectors
        feedforward_dim: Dimension of the feedforward network.
            If None, uses embed_dim
        p_dropout: Dropout probability
        activation: Activation function for the feedforward network.
            Defaults to ReLU
        norm_first: If True, applies normalization before attention (Pre-LN),
            else applies it after (Post-LN)
        **kwargs: Additional keyword arguments passed to the attention mechanism
    """

    def __init__(
        self,
        attention: BaseMultiHeadAttention,
        embed_dim: int,
        feedforward_dim: int | None = None,
        p_dropout: float = 0.0,
        activation: nn.Module | None = None,
        norm_first: bool = False,
    ):
        super().__init__()
        self.attn = attention
        self.embed_dim = embed_dim

        feedforward_dim = embed_dim if feedforward_dim is None else feedforward_dim
        self.feedforward_dim = feedforward_dim
        
        activation = nn.ReLU() if activation is None else copy.deepcopy(activation)

        # Feedforward model.
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation,
            nn.Dropout(p_dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(p_dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first

        self.attn_dropout = nn.Dropout(p_dropout)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method to be implemented by derived classes.
        Defines the forward pass through the layer.
        """
        pass
