import warnings

import torch

from ...utils.helpers import get_clones
from ..attention_layers import MultiHeadAttentionLayer
from .base import BaseTransformerEncoder


class TransformerEncoder(BaseTransformerEncoder):
    """
    ACNPs/TNPs Transformer Encoder that applies self-attention to all inputs.
    """

    def __init__(
        self,
        layer: MultiHeadAttentionLayer,
        num_layers: int,
    ):
        """
        Initialize the transformer encoder.

        Args:
            layer (MultiHeadAttentionLayer): Attention layer to be cloned.
            num_layers (int): Number of transformer layers.
        """
        super().__init__(num_layers=num_layers)
        self.layers = get_clones(layer, num_layers)

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            z = layer(zq=z, zk=z, zv=z, mask=mask)

        return z


class EfficientQueryTransformerEncoder(BaseTransformerEncoder):
    """
    Efficient transformer encoder for TNPs where context and query inputs
    are processed through separate paths, potentially with shared parameters.
    Based on https://arxiv.org/pdf/2211.08458
    """

    def __init__(
        self,
        layer: MultiHeadAttentionLayer,
        num_layers: int,
        share_params: bool = True,
    ):
        """
        Initialize the efficient query transformer encoder.

        Args:
            layer (MultiHeadAttentionLayer): Attention layer to clone.
            num_layers (int): Number of encoder layers.
            share_params (bool): Whether to share parameters between context
                                 and query encoders.
        """
        super().__init__(num_layers=num_layers)

        self.context_to_context_layers = get_clones(layer, num_layers)
        self.context_to_query_layers = (
            self.context_to_context_layers
            if share_params
            else get_clones(layer, num_layers)
        )

    def forward(
        self,
        zc: torch.Tensor,
        zq: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn(
                "The mask will be ignored in the Efficient-Query TNP Encoder."
            )
        for context_to_context_layer, context_to_query_layer in zip(
            self.context_to_context_layers, self.context_to_query_layers
        ):
            zc = context_to_context_layer(zq=zc, zk=zc, zv=zc)
            zq = context_to_query_layer(zq=zq, zk=zc, zv=zc)

        return zq
