import torch

from .base import BaseMultiHeadAttentionLayer


class MultiHeadAttentionLayer(BaseMultiHeadAttentionLayer):
    """
    Standard multi-head attention layer with residual connections
    and layer normalization.

    Implements the standard Transformer encoder/decoder layer architecture
    with multi-head attention followed by a feedforward network.
    """

    def _attn_block(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute attention and apply dropout.

        Args:
            zq: Query embeddings [batch_size, num_query, embed_dim]
            zk: Key embeddings [batch_size, num_key, embed_dim]
            zv: Value embeddings [batch_size, num_value, embed_dim]
            mask: Optional attention mask [batch_size, num_query, num_key]

        Returns:
            Tensor: Updated query embeddings after attention
        """
        x = self.attn(zq, zk, zv, mask=mask)
        return self.attn_dropout(x)

    def forward(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the attention layer.

        Args:
            zq: Query embeddings [batch_size, num_query, embed_dim]
            zk: Key embeddings [batch_size, num_key, embed_dim]
            zv: Value embeddings [batch_size, num_value, embed_dim]
            mask: Optional attention mask [batch_size, num_query, num_key]

        Returns:
            Tensor: Updated query embeddings
        """
        if self.norm_first:
            zq_norm = self.norm1(zq)
            zk_norm = zq_norm if zk is zq else self.norm1(zk)
            zv_norm = zk_norm if zv is zk else self.norm1(zv)

            zq = zq + self._attn_block(zq_norm, zk_norm, zv_norm, mask)
            zq = zq + self.ff_block(self.norm2(zq))
        else:
            zq = self.norm1(zq + self._attn_block(zq, zk, zv, mask))
            zq = self.norm2(zq + self.ff_block(zq))

        return zq
