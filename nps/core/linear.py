import einops
import torch
import torch.nn as nn


class GroupLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        groups = groups or 1
        assert in_features % groups == 0, "in_features must be divisible by groups"
        assert out_features % groups == 0, "out_features must be divisible by groups"

        self.groups = groups
        self.in_feat_per_group = in_features // groups
        self.out_feat_per_group = out_features // groups

        self.weight = nn.Parameter(
            torch.randn(groups, self.in_feat_per_group, self.out_feat_per_group)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(groups, self.out_feat_per_group))
        else:
            self.bias = None

    def forward(self, x):
        x = einops.rearrange(
            x, "... (g c) -> ... g c", g=self.groups, c=self.in_feat_per_group
        )  # [B, G, in_feat_per_group]
        out = torch.einsum(
            "...gi,gio->...go", x, self.weight
        )  # [B, G, out_feat_per_group]
        if self.bias is not None:
            out += self.bias
        out = einops.rearrange(
            out,
            "... g c -> ... (g c)",
        )
        return out
