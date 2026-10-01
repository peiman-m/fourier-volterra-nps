from collections.abc import Callable

import einops
import torch
import torch.nn as nn

from ...utils.group_actions import translation
from .base import BaseMultiHeadAttention


class MultiHeadTEAttention(BaseMultiHeadAttention):
    def __init__(
        self,
        *,
        kernel: nn.Module,
        group_action: Callable = translation,
        phi: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel
        self.group_action = group_action  # Group action on inputs prior to kernel.
        self.phi = phi  # Additional transformation on spatio-temporal locations.
        self._build()

    def _build(self) -> None:
        inner_dim = self.head_dim * self.num_heads
        project_out = not (self.num_heads == 1 and self.head_dim == self.v_dim)

        self.to_q = nn.Linear(self.q_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(self.k_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(self.v_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, self.v_dim), nn.Dropout(self.p_dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes multi-head translation equivariant attention.

        Args:
            zq (torch.Tensor): Query token.
            zk (torch.Tensor): Key token.
            zv (torch.Tensor): Value token.
            xq (torch.Tensor): Query input locations.
            xkv (torch.Tensor): Key input locations.
            mask (torch.Tensor | None, optional): Query-key mask. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output of attention mechanism.
        """
        # Compute output of group action.
        # (m, nq, nkv, dx).
        diff = self.group_action(xq, xkv)

        # Compute token attention.
        q = self.to_q(zq)
        k = self.to_k(zk)
        v = self.to_v(zv)

        # Each of shape (m, {num_heads, qk_dim}, n, head_dim).
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q, k, v),
        )

        # (m, h, nq, nkv).
        token_dots = (q @ k.transpose(-1, -2)) * self.scale
        token_dots = einops.rearrange(token_dots, "b h nq nkv -> b nq nkv h")
        kernel_input = torch.cat((diff, token_dots), dim=-1)
        dots = self.kernel(kernel_input)
        dots = einops.rearrange(dots, "b nq nkv h -> b h nq nkv")

        if mask is not None:
            # Follow scaled_dot_product_attention's convention (as MultiHeadAttention
            # does), so both attention modules accept the same masks. Applied
            # post-kernel so masked pairs reach softmax as -inf regardless of the
            # kernel's output.
            mask = einops.repeat(mask, "b nq nkv -> b h nq nkv", h=self.num_heads)
            if mask.dtype == torch.bool:
                # Boolean: True = keep, False = drop.
                dots = dots.masked_fill(~mask, -float("inf"))
            else:
                # Float: additive bias (0 keep, -inf drop), as the encoders build.
                dots = dots + mask

        # (m, num_heads, nq, nkv).
        attn = dots.softmax(dim=-1)

        out = attn @ v
        out = einops.rearrange(out, "b h nq d -> b nq (h d)")
        out = self.to_out(out)

        # Also update spatio-temporal locations if necessary.
        if self.phi:
            phi_input = einops.rearrange(attn, "b h nq nkv -> b nq nkv h")
            x_dots = self.phi(phi_input)
            xq_new = xq + (diff * x_dots).mean(-2)
        else:
            xq_new = xq

        return out, xq_new
