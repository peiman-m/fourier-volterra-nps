import torch

from ...utils.helpers import get_clones
from ..attention_layers import MultiHeadTEAttentionLayer
from .base import BaseTransformerEncoder


class TETransformerEncoder(BaseTransformerEncoder):
    """
    TETNPs Transformer Encoder that applies translation-equivariant
    self-attention to all inputs.
    """

    def __init__(
        self,
        layer: MultiHeadTEAttentionLayer,
        num_layers: int,
    ):
        """
        Initialize the translation-equivariant transformer encoder.

        Args:
            layer (MultiHeadTEAttentionLayer): Attention layer to be cloned.
            num_layers (int): Number of transformer layers.
        """
        super().__init__(num_layers=num_layers)
        self.layers = get_clones(layer, num_layers)

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        for layer in self.layers:
            z, _ = layer(zq=z, zk=z, zv=z, xq=x, xkv=x, mask=mask)

        return z


class TEEfficientQueryTransformerEncoder(BaseTransformerEncoder):
    """
    Efficient transformer encoder for TETNPs where context and query inputs
    are processed through separate paths, potentially with shared parameters.
    Based on https://arxiv.org/pdf/2406.12409
    """

    def __init__(
        self,
        layer: MultiHeadTEAttentionLayer,
        num_layers: int,
        share_params: bool = True,
    ):
        """
        Initialize the efficient query translation-equivariant transformer encoder.

        Args:
            layer (MultiHeadTEAttentionLayer): Attention layer to clone.
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
        xc: torch.Tensor,
        xq: torch.Tensor,
    ) -> torch.Tensor:
        for context_to_context_layer, context_to_query_layer in zip(
            self.context_to_context_layers, self.context_to_query_layers
        ):
            zc, _ = context_to_context_layer(zq=zc, zk=zc, zv=zc, xq=xc, xkv=xc)
            zq, _ = context_to_query_layer(zq=zq, zk=zc, zv=zc, xq=xq, xkv=xc)

        return zq
