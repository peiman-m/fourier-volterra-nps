import warnings
from typing import Literal, cast

import einops
import torch
import torch.nn as nn

from ..core.attentions.attention import MultiHeadAttention

ReductionType = (
    Literal["sum"]
    | Literal["mean"]
    | Literal["min"]
    | Literal["max"]
    | Literal["quantile"]
)
ReductionArg = ReductionType | list[ReductionType]


class Aggregator:
    """
    Compute one or more reductions (sum, mean, min, max, quantile)
    over the specified dimension of a tensor, optionally masking out
    elements before reduction. All reductions ignore NaNs.

    Args:
        reduction: A single reduction name or a list of names.
                   Valid options: "sum", "mean", "min", "max", "quantile".
        quantiles: If "quantile" is requested, provide a float or 1D list
                   of floats in [0.0, 1.0]. Ignored otherwise.
        dim:       Dimension along which to reduce.
        keepdim:   Whether to keep the reduced dimension.
    """

    # Move function map to class attribute to avoid rebuilding each __init__
    _FN_MAP = {
        "sum": "_nansum",
        "mean": "_nanmean",
        "min": "_nanmin",
        "max": "_nanmax",
        "quantile": "_nanquantile",
    }

    def __init__(
        self,
        reduction: ReductionArg = "mean",
        quantiles: float | list[float] | torch.Tensor | None = None,
        dim: int = 1,
        keepdim: bool = False,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim

        # 1) Normalize and deduplicate reductions
        reductions_list = self._normalize_reduction_arg(reduction)
        if len(reductions_list) < (
            1
            if isinstance(reduction, str)
            else len(reduction)  # type: ignore[arg-type]
        ):
            warnings.warn(
                "Duplicate reductions removed. " f"Using: {reductions_list!r}"
            )

        # 2) Validate reductions against allowed keys
        for r in reductions_list:
            if r not in self._FN_MAP:
                raise ValueError(
                    f"Invalid reduction '{r}'. " f"Choose from {set(self._FN_MAP)}."
                )
        self.reductions = reductions_list

        # 3) Handle quantile setup (if requested)
        if "quantile" in self.reductions:
            if quantiles is None:
                raise ValueError(
                    "`quantiles` must be provided when " "'quantile' is in reductions."
                )
            self.quantiles = self._prepare_quantiles_tensor(quantiles)
        else:
            if quantiles is not None:
                warnings.warn(
                    "`quantiles` provided but 'quantile' not in reductions; "
                    "ignoring."
                )
            self.quantiles = None

    @staticmethod
    def _normalize_reduction_arg(reduction: ReductionArg) -> list[ReductionType]:
        """
        Turn a single string or list-of-strings into a lowercase list
        with duplicates removed (preserving order).
        """
        if isinstance(reduction, str):
            items = [reduction.lower()]
        elif isinstance(reduction, list):
            items = [r.lower() for r in reduction]
        else:
            raise TypeError(f"Expected str or list[str], got {type(reduction)}.")

        # Deduplicate while preserving order
        seen: dict[str, None] = {}
        for r in items:
            seen[r] = None
        # Validity of each reduction name is checked downstream.
        return cast("list[ReductionType]", list(seen.keys()))

    @staticmethod
    def _prepare_quantiles_tensor(
        quantiles: float | list[float] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert float / list / 1D tensor into a 1D float tensor of unique,
        sorted quantiles in [0,1].
        """
        # Convert to 1D float-tensor
        if isinstance(quantiles, torch.Tensor):
            if quantiles.ndim != 1:
                raise ValueError(
                    "`quantiles` tensor must be 1D, " f"got {quantiles.ndim}D."
                )
            q = quantiles.to(torch.float)
        elif isinstance(quantiles, (float, int)):
            q = torch.tensor([float(quantiles)], dtype=torch.float)
        elif isinstance(quantiles, list):
            q = torch.tensor(list(quantiles), dtype=torch.float)
        else:
            raise TypeError(f"Expected float/list/torch.Tensor, got {type(quantiles)}.")

        if q.ndim != 1:
            raise ValueError(
                "`quantiles` must be a float or 1D list, "
                f"got shape {tuple(q.shape)}."
            )
        if torch.any((q < 0.0) | (q > 1.0)):
            raise ValueError("All quantiles must be within [0.0, 1.0].")

        # Deduplicate and sort
        q_unique = torch.sort(torch.unique(q)).values
        if q_unique.numel() < q.numel():
            warnings.warn(
                "Duplicate quantile levels removed. " f"Using: {q_unique.tolist()!r}"
            )
        return q_unique

    def _apply_mask(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            return x

        if mask.dtype != torch.bool:
            raise TypeError(
                "`mask` must have dtype torch.bool, " f"but got {mask.dtype}."
            )

        if mask.shape != x.shape[:-1]:
            raise ValueError(
                f"`mask` must match the spatial shape of `x` "
                "(excluding the last dimension), "
                f"but got mask.shape={mask.shape}, x.shape={x.shape}."
            )

        # Expand mask to match x’s last dimension
        expanded = einops.repeat(mask, "... -> ... d", d=x.shape[-1])
        return x.masked_fill(expanded, torch.nan)

    def _nansum(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nansum(x, dim=self.dim, keepdim=self.keepdim)

    def _nanmean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(x, dim=self.dim, keepdim=self.keepdim)

    def _nanquantile(self, x: torch.Tensor, q: float | torch.Tensor) -> torch.Tensor:
        """
        Compute nanquantile at level q (float or 1D tensor).
        If q is 1D, returns a tensor whose dimension at 'dim'
        is expanded to include the quantile axis.
        """
        if isinstance(q, float):
            # Single quantile → direct call
            return torch.nanquantile(x, q, dim=self.dim, keepdim=self.keepdim)

        if isinstance(q, torch.Tensor):
            if q.ndim != 1:
                raise ValueError(f"Tensor of quantiles must be 1D, got {q.ndim}D.")
            # Compute all quantiles at once: result shape = [nq, *, ...]
            raw = torch.nanquantile(
                x, q.to(x.device), dim=self.dim, keepdim=self.keepdim
            )
            return self._reshape_multi_quantiles(raw)

        raise TypeError(f"Expected float or 1D torch.Tensor for 'q', got {type(q)}.")

    def _nanmin(self, x: torch.Tensor) -> torch.Tensor:
        # min is quantile at 0.0
        return self._nanquantile(x, 0.0)

    def _nanmax(self, x: torch.Tensor) -> torch.Tensor:
        # max is quantile at 1.0
        return self._nanquantile(x, 1.0)

    def _reshape_multi_quantiles(self, qvals: torch.Tensor) -> torch.Tensor:

        if self.keepdim:
            nq, *axis, d = qvals.shape

            # Build axis labels
            axis_labels = [f"n{i}" for i in range(len(axis))]
            pattern_in = f"nq {' '.join(axis_labels)} d"

            # pattern: (nq n0 n1 ... nk-1 d) ->
            # (n0 ... n{m-1} (nq n{m}) n{m+1} ... nk-1 d)
            axis_labels[self.dim] = f"(nq {axis_labels[self.dim]})"
            pattern_out = f"{' '.join(axis_labels)} d"
            return einops.rearrange(qvals, f"{pattern_in} -> {pattern_out}")
        else:
            # pattern: (nq ... d) -> (... (nq d))
            return einops.rearrange(qvals, "nq ... d -> ... (nq d)")

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply the requested reductions to x (shape [..., D]),
        optionally masking out entries before reducing. Returns
        either a single Tensor or a concatenation across the
        feature-dimension if multiple reductions are requested.

        Args:
            x:    Input tensor of shape [B, N, D] (or more dims,
                    but reduction happens on dim=self.dim).
            mask: Optional boolean mask of shape [B, N];
                    True = mask out that entry.

        Returns:
            If only one reduction was requested,
                returns a Tensor of shape [..., D'].
            If multiple reductions were requested,
                returns a concatenated Tensor
            along the feature dimension (axis=self.dim)
                with shape [..., D_total].
        """
        x_masked = self._apply_mask(x, mask)

        outputs: list[torch.Tensor] = []
        for r in self.reductions:
            fn = getattr(self, self._FN_MAP[r])
            if r == "quantile":
                out = fn(x_masked, self.quantiles)  # type: ignore[arg-type]
            else:
                out = fn(x_masked)  # type: ignore[arg-type]
            outputs.append(out)

        if len(outputs) == 1:
            return outputs[0]

        # If multiple outputs, concatenate along the reduced dimension
        return torch.cat(outputs, dim=self.dim)


class PMAAggregator(nn.Module):
    """Pooling by Multihead Attention (Set Transformer, Lee et al. 2019).

    Attention-based set pooling: learnable seed vectors act as queries over
    the set elements. Drop-in replacement for ``Aggregator`` with
    ``reduction="mean"`` when ``num_seeds=1``; output shape is ``(B, D)``.
    With ``num_seeds > 1``, output flattens to ``(B, num_seeds * D)``, so
    downstream decoders must size their ``in_dim`` accordingly.

    Mask convention matches ``Aggregator``: ``True`` means "mask out"
    (padding). Internally the mask is inverted to SDPA's ``True`` = attend
    convention.

    Args:
        embed_dim: Width of the input set elements' feature dim.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension. Defaults to ``embed_dim // num_heads``.
        num_seeds: Number of learnable seed queries (k in the paper).
        seed_init_scale: Std-dev of the N(0, sigma^2) init for seed vectors.
        p_dropout: Attention dropout.
    """

    name: str = "pma"

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        head_dim: int | None = None,
        num_seeds: int = 1,
        seed_init_scale: float = 0.02,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim={embed_dim} not divisible by "
                    f"num_heads={num_heads}; pass head_dim explicitly."
                )
            head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_seeds = num_seeds

        self.seeds = nn.Parameter(
            torch.randn(1, num_seeds, embed_dim) * seed_init_scale
        )
        self.attention = MultiHeadAttention(
            q_dim=embed_dim,
            k_dim=embed_dim,
            v_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            p_dropout=p_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention-pool the set.

        Args:
            x: Input tensor of shape ``(B, N, D)``.
            mask: Optional boolean mask of shape ``(B, N)`` where ``True``
                indicates padding (to be excluded from attention).

        Returns:
            Tensor of shape ``(B, D)`` when ``num_seeds == 1``, else
            ``(B, num_seeds * D)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"PMAAggregator expects (B, N, D); got shape {tuple(x.shape)}."
            )
        batch_size = x.size(0)
        seeds = self.seeds.expand(batch_size, -1, -1)

        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError(
                    f"mask must be torch.bool, got {mask.dtype}."
                )
            if mask.shape != x.shape[:-1]:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match x[:, :, :-1] "
                    f"shape {tuple(x.shape[:-1])}."
                )
            # Aggregator: True = padding. MultiHeadAttention expects True =
            # attend. Invert + add query dim -> broadcast over all seed queries.
            attn_mask = (~mask).unsqueeze(1)
        else:
            attn_mask = None

        out = self.attention(seeds, x, x, mask=attn_mask)

        if self.num_seeds == 1:
            return out.squeeze(1)
        return out.flatten(1)
