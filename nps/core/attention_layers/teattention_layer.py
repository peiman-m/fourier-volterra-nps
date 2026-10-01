import torch

from .base import BaseMultiHeadAttentionLayer


class MultiHeadTEAttentionLayer(BaseMultiHeadAttentionLayer):
    """
    Translation equivariant multi-head attention layer.

    This layer implements attention that is equivariant to translations,
    which means the output transforms predictably when the input is translated.
    """

    def _attn_block(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute translation equivariant attention and apply dropout.

        Args:
            zq: Query feature embeddings [batch_size, num_query, embed_dim]
            zk: Key feature embeddings [batch_size, num_key, embed_dim]
            zv: Value feature embeddings [batch_size, num_value, embed_dim]
            xq: Query position embeddings [batch_size, num_query, position_dim]
            xkv: Key/value position embeddings [batch_size, num_key/value, position_dim]
            mask: Optional attention mask [batch_size, num_query, num_key]

        Returns:
            Tuple[Tensor, Tensor]: Updated feature embeddings and position embeddings
        """
        zq, xq = self.attn(zq, zk, zv, xq, xkv, mask=mask)
        return self.attn_dropout(zq), xq

    def forward(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the translation equivariant attention layer.

        Args:
            zq: Query feature embeddings [batch_size, num_query, embed_dim]
            zk: Key feature embeddings [batch_size, num_key, embed_dim]
            zv: Value feature embeddings [batch_size, num_value, embed_dim]
            xq: Query position embeddings [batch_size, num_query, position_dim]
            xkv: Key/value position embeddings [batch_size, num_key/value, position_dim]
            mask: Optional attention mask [batch_size, num_query, num_key]

        Returns:
            Tuple[Tensor, Tensor]: Updated feature embeddings and position embeddings
        """
        if self.norm_first:
            zq_norm = self.norm1(zq)
            zk_norm = zq_norm if zk is zq else self.norm1(zk)
            zv_norm = zk_norm if zv is zk else self.norm1(zv)

            zq_update, xq = self._attn_block(
                zq_norm, zk_norm, zv_norm, xq, xkv, mask
            )
            zq = zq + zq_update
            zq = zq + self.ff_block(self.norm2(zq))
        else:
            zq_update, xq = self._attn_block(
                zq, zk, zv, xq, xkv, mask
            )
            zq = self.norm1(zq + zq_update)
            zq = self.norm2(zq + self.ff_block(zq))

        return zq, xq
