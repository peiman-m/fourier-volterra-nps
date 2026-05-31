import einops
import torch

from ...utils.helpers import get_clones, warn_once
from ..attention_layers import MultiHeadAttentionLayer
from .base import BaseTransformerEncoder


class ISTransformerEncoder(BaseTransformerEncoder):
    """
    Induced Set Transformer (IST) encoder that uses a fixed size set of learnable
    pseudo tokens to process context data and generate query representations.
    Unlike Perceiver, IST also updates the context representations through cross attention
    among pseudo tokens and context points.
    """

    def __init__(
        self,
        layer: MultiHeadAttentionLayer,
        num_layers: int,
        num_pseudo_tokens: int,
    ):
        """
        Initialize the IST transformer encoder.

        Args:
            layer (MultiHeadAttentionLayer): Attention layer to clone.
            num_layers (int): Number of encoder layers.
            num_pseudo_tokens (int): Number of pseudo tokens.
        """
        super().__init__(num_layers=num_layers)

        self.num_pseudo_tokens = num_pseudo_tokens

        # Initialize learnable pseudo tokens
        embed_dim = layer.embed_dim
        self.z_pt = torch.nn.Parameter(torch.randn(num_pseudo_tokens, embed_dim))

        # Clone layers - note that pseudo_to_context has one fewer layer than the others
        self.context_to_pseudo_layers = get_clones(layer, num_layers)
        self.pseudo_to_context_layers = get_clones(layer, num_layers - 1)
        self.pseudo_to_query_layers = get_clones(layer, num_layers)

    def forward(
        self,
        zc: torch.Tensor,
        zq: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            warn_once(
                "mask is not currently being used in ISTransformerEncoder."
            )

        # Expand z_pt for batch
        batch_size = zc.shape[0]
        z_pt = einops.repeat(self.z_pt, "n e -> b n e", b=batch_size)

        for context_to_pseudo_layer, pseudo_to_context_layer, pseudo_to_query_layer in zip(
            self.context_to_pseudo_layers[:-1],
            self.pseudo_to_context_layers,
            self.pseudo_to_query_layers[:-1],
        ):
            # Cross-attention: pseudo tokens attend to context
            z_pt = context_to_pseudo_layer(zq=z_pt, zk=zc, zv=zc)

            # Cross-attention: contexts attend to pseudo tokens (skip in final layer)
            zc = pseudo_to_context_layer(zq=zc, zk=z_pt, zv=z_pt)

            # Cross-attention: queries attend to pseudo tokens
            zq = pseudo_to_query_layer(zq=zq, zk=z_pt, zv=z_pt)

        z_pt = self.context_to_pseudo_layers[-1](zq=z_pt, zk=zc, zv=zc)
        zq = self.pseudo_to_query_layers[-1](zq=zq, zk=z_pt, zv=z_pt)

        return zq
