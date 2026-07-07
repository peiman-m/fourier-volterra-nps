import einops
import torch

from ...utils.helpers import get_clones
from ..attention_layers import MultiHeadAttentionLayer
from .base import BaseTransformerEncoder


class PerceiverEncoder(BaseTransformerEncoder):
    """
    Perceiver-based transformer encoder that uses a fixed-size set of learnable
    pseudo tokens to attend to context data and then generate query representations.
    """

    def __init__(
        self,
        layer: MultiHeadAttentionLayer,
        num_layers: int,
        num_pseudo_tokens: int,
    ):
        """
        Initialize the perceiver transformer encoder.

        Args:
            layer (MultiHeadAttentionLayer): Attention layer to clone.
            num_layers (int): Number of transformer layers.
            num_pseudo_tokens (int): Number of pseudo tokens.
        """
        super().__init__(num_layers=num_layers)

        self.num_pseudo_tokens = num_pseudo_tokens

        # Initialize learnable pseudo tokens
        embed_dim = layer.embed_dim
        self.z_pt = torch.nn.Parameter(torch.randn(num_pseudo_tokens, embed_dim))

        # Clone layers
        self.context_to_pseudo_layers = get_clones(layer, num_layers)
        self.pseudo_to_pseudo_layers = get_clones(layer, num_layers)
        self.pseudo_to_query_layers = get_clones(layer, num_layers)

    def forward(
        self,
        zc: torch.Tensor,
        zq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the perceiver encoder.

        Args:
            zc: Context representations [batch_size, num_context, embed_dim]
            zq: Query representations [batch_size, num_query, embed_dim]

        Returns:
            Enhanced query representations [batch_size, num_query, embed_dim]
        """
        # Expand pseudo tokens for batch
        batch_size = zc.shape[0]
        z_pt = einops.repeat(self.z_pt, "n e -> b n e", b=batch_size)

        for context_to_pseudo_layer, pseudo_to_pseudo_layer, pseudo_to_query_layer in zip(
            self.context_to_pseudo_layers,
            self.pseudo_to_pseudo_layers,
            self.pseudo_to_query_layers,
        ):
            # Cross-attention: pseudo tokens attend to context
            z_pt = context_to_pseudo_layer(zq=z_pt, zk=zc, zv=zc)

            # Self-attention among pseudo tokens
            z_pt = pseudo_to_pseudo_layer(zq=z_pt, zk=z_pt, zv=z_pt)

            # Cross-attention: queries attend to pseudo tokens
            zq = pseudo_to_query_layer(zq=zq, zk=z_pt, zv=z_pt)

        return zq
