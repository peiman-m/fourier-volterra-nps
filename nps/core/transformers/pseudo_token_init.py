import einops
import torch
from torch import nn


class PseudoTokenInitialiser(nn.Module):
    """
    Pseudo-token initializer for translation-equivariant models.
    Initializes latent locations based on attention-weighted combinations 
        of context locations.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        p_dropout: float = 0.0,
    ):
        """
        Initialize the pseudo-token initializer.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            head_dim (int): Dimension per head
            p_dropout (float): Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == embed_dim)

        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(p_dropout) if project_out else nn.Identity(),
        )

        # Pre-softmax weighting of location attention weights.
        self.raw_head_weights = nn.Parameter(torch.ones((num_heads,)))

    @property
    def head_weights(self):
        """Normalized attention head weights."""
        return self.raw_head_weights.softmax(dim=0)

    def forward(
        self,
        zq: torch.Tensor,
        zkv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize pseudo-tokens based on context.

        Args:
            zq: Query tokens (latent tokens)
            zkv: Key/Value tokens (context tokens)
            xq: Query locations (latent locations)
            xkv: Key/Value locations (context locations)

        Returns:
            tuple of (unchanged tokens, updated locations)
        """
        q = self.to_q(zq)
        k = self.to_k(zkv)

        # Each of shape (b, num_heads, n, head_dim).
        q, k = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q, k),
        )

        xkv_ = einops.repeat(xkv, "b n d -> b h n d", h=self.num_heads)

        xq_update = (
            nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                q, k, xkv_, scale=self.scale
            )
        )

        # Now do weighted sum over heads.
        xq_update = einops.rearrange(xq_update, "b h n d -> b n d h")
        xq_update = xq_update @ self.head_weights

        xq_out = xq + xq_update

        return zq, xq_out
