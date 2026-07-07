import einops
import torch

from ...utils.helpers import get_clones, warn_once
from ..attention_layers import MultiHeadTEAttentionLayer
from .base import BaseTransformerEncoder
from .pseudo_token_init import PseudoTokenInitialiser


class TEPerceiverEncoder(BaseTransformerEncoder):
    """
    Translation-equivariant perceiver-based transformer encoder that uses a fixed
    size set of latent pseudo tokens to attend to context data and then generate query
    representations while maintaining translation equivariance.
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
        Initialize the translation-equivariant perceiver transformer encoder.

        Args:
            layer (MultiHeadTEAttentionLayer): Multihead TE attention layer to clone.
            num_pseudo_tokens (int): Number of pseudo tokens.
            num_layers (int): Number of transformer layers.
            x_dim (int): Spatial dimension of inputs.
            pseudo_token_initialiser (PseudoTokenInitialiser | None): Optional initializer
                for latent locations.
        """
        super().__init__(num_layers=num_layers)

        self.num_pseudo_tokens = num_pseudo_tokens
        self.x_dim = x_dim

        # Initialize learnable pseudo tokens (tokens and locations)
        embed_dim = layer.embed_dim
        self.z_pt = torch.nn.Parameter(torch.randn(num_pseudo_tokens, embed_dim))
        self.x_pt = torch.nn.Parameter(torch.randn(num_pseudo_tokens, self.x_dim))

        # Clone layers
        self.context_to_pseudo_layers = get_clones(layer, num_layers)
        self.pseudo_to_pseudo_layers = get_clones(layer, num_layers)
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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the translation-equivariant perceiver encoder.

        Args:
            zc: Context token representations [batch_size, num_context, embed_dim]
            zq: Query token representations [batch_size, num_query, embed_dim]
            xc: Context spatial locations [batch_size, num_context, dim]
            xq: Query spatial locations [batch_size, num_query, dim]
            mask: Optional attention mask

        Returns:
            Enhanced query representations [batch_size, num_query, embed_dim]
        """
        if mask is not None:
            warn_once(
                "mask is not currently being used in TEPerceiverEncoder."
            )

        # Expand pseudo tokens for batch
        batch_size = zc.shape[0]
        z_pt = einops.repeat(self.z_pt, "n e -> b n e", b=batch_size)
        x_pt = einops.repeat(self.x_pt, "n x -> b n x", b=batch_size)

        # Initialize pseudo-tokens
        z_pt, x_pt = self.pseudo_token_initialiser(
            z_pt, zc, x_pt, xc
        )

        for context_to_pseudo_layer, pseudo_to_pseudo_layer, pseudo_to_query_layer in zip(
            self.context_to_pseudo_layers,
            self.pseudo_to_pseudo_layers,
            self.pseudo_to_query_layers,
        ):
            # Cross-attention: pseudo tokens attend to context
            z_pt, x_pt = context_to_pseudo_layer(
                zq=z_pt, zk=zc, zv=zc, xq=x_pt, xkv=xc
            )

            # Self-attention among pseudo tokens
            z_pt, x_pt = pseudo_to_pseudo_layer(
                zq=z_pt, zk=z_pt, zv=z_pt, xq=x_pt, xkv=x_pt,
            )

            # Cross-attention: queries attend to pseudo tokens
            zq, xq = pseudo_to_query_layer(
                zq=zq, zk=z_pt, zv=z_pt, xq=xq, xkv=x_pt,
            )

        return zq
