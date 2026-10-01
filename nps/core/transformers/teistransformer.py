import einops
import torch

from ...utils.helpers import get_clones
from ..attention_layers import MultiHeadTEAttentionLayer
from .base import BaseTransformerEncoder
from .pseudo_token_init import PseudoTokenInitialiser


class TEISTransformerEncoder(BaseTransformerEncoder):
    """
    Translation-equivariant Induced Set Transformer (IST) encoder that uses a fixed
    size set of pseudo tokens to process context data and generate query representations
    while maintaining translation equivariance. Unlike TE Perceiver, TEIST also updates
    the context representations through cross attention among pseudo tokens and context points.
    """

    def __init__(
        self,
        layer: MultiHeadTEAttentionLayer,
        num_layers: int,
        num_pseudo_tokens: int,
        x_dim: int,
        pseudo_token_initialiser: PseudoTokenInitialiser | None = None,
    ):
        """
        Initialize the translation-equivariant IST transformer encoder.

        Args:
            layer (MultiHeadTEAttentionLayer): Multihead TE attention layer to clone.
            num_layers (int): Number of transformer layers.
            num_pseudo_tokens (int): Number of pseudo tokens.
            x_dim (int): Spatial dimension of inputs.
            pseudo_token_initialiser (PseudoTokenInitialiser | None):
                Optional initializer for pseudo tokens locations.
        """
        super().__init__(num_layers=num_layers)

        self.num_pseudo_tokens = num_pseudo_tokens
        self.x_dim = x_dim

        # Initialize learnable pseudo tokens (tokens and locations)
        embed_dim = layer.embed_dim
        self.z_pt = torch.nn.Parameter(torch.randn(num_pseudo_tokens, embed_dim))
        self.x_pt = torch.nn.Parameter(torch.randn(num_pseudo_tokens, x_dim))

        # Clone layers - all have same number of layers unlike standard IST
        self.context_to_pseudo_layers = get_clones(layer, num_layers)
        self.pseudo_to_context_layers = get_clones(layer, num_layers - 1)
        self.pseudo_to_query_layers = get_clones(layer, num_layers)

        # Set up pseudo-token initializer
        if pseudo_token_initialiser is None:
            self.pseudo_token_initialiser = lambda z_pt, zc, x_pt, xc: (
                z_pt,
                x_pt + xc.mean(-2, keepdim=True),
            )
        else:
            self.pseudo_token_initialiser = pseudo_token_initialiser

    def forward(
        self,
        zc: torch.Tensor,
        zq: torch.Tensor,
        xc: torch.Tensor,
        xq: torch.Tensor,
    ) -> torch.Tensor:
        # Expand latents for batch
        batch_size = zc.shape[0]
        z_pt = einops.repeat(self.z_pt, "n e -> b n e", b=batch_size)
        x_pt = einops.repeat(self.x_pt, "n x -> b n x", b=batch_size)

        # Initialize pseudo-tokens
        z_pt, x_pt = self.pseudo_token_initialiser(
            z_pt, zc, x_pt, xc
        )

        for context_to_pseudo_layer, pseudo_to_context_layer, pseudo_to_query_layer in zip(
            self.context_to_pseudo_layers[:-1],
            self.pseudo_to_context_layers,
            self.pseudo_to_query_layers[:-1],
        ):
            # Cross-attention: pseudo tokens attend to context
            z_pt, x_pt = context_to_pseudo_layer(
                zq=z_pt, zk=zc, zv=zc, xq=x_pt, xkv=xc
            )

            # Cross-attention: contexts attend to pseudo tokens (skip in final layer)
            zc, xc = pseudo_to_context_layer(
                zq=zc, zk=z_pt, zv=z_pt, xq=xc, xkv=x_pt,
            )

            # Cross-attention: queries attend to pseudo tokens
            zq, xq = pseudo_to_query_layer(
                zq=zq, zk=z_pt, zv=z_pt, xq=xq, xkv=x_pt,
            )

        # Final context-to-latent attention and latent-to-query attention
        z_pt, x_pt = self.context_to_pseudo_layers[-1](
            zq=z_pt, zk=zc, zv=zc, xq=x_pt, xkv=xc
        )
        zq, _ = self.pseudo_to_query_layers[-1](
            zq=zq, zk=z_pt, zv=z_pt, xq=xq, xkv=x_pt
        )

        return zq
