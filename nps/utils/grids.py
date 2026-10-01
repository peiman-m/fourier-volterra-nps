from collections.abc import Callable
from typing import Any

import einops
import torch


def flatten_grid(
    x: torch.Tensor,
    start_dim: int = 1,
    end_dim: int = -1,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Flattens a multi-dimensional grid tensor into a single dimension
    and provides functions for conversion.

    Args:
        x (torch.Tensor): The input tensor with a grid-like structure.
        start_dim (int, optional): The first dimension to include in flattening.
            Defaults to 1.
        end_dim (int, optional): The last dimension to include in flattening.
            Defaults to -1.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The flattened tensor.
            - Callable[[torch.Tensor], torch.Tensor]: A function that converts
                flattened tensors back to grid structure.
    """
    grid_shape = x.shape[start_dim:end_dim]
    n_strings = [f"n{i}" for i in range(len(grid_shape))]
    grid_pattern = f"b {' '.join(n_strings)} e"
    flat_pattern = f"b ({' '.join(n_strings)}) e"
    grid_to_flat = grid_pattern + " -> " + flat_pattern
    flat_to_grid = flat_pattern + " -> " + grid_pattern
    reshape_vars = dict(zip(n_strings, grid_shape))

    def grid_to_flat_fn(x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, grid_to_flat)

    def flat_to_grid_fn(x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, flat_to_grid, **reshape_vars)

    return grid_to_flat_fn(x), flat_to_grid_fn


def construct_grid(
    span: (
        tuple[tuple[float, float], ...]  # dim * [min, max]
        | tuple[float, float]  # [min, max] for all dimensions
        | torch.Tensor  # shape [d, 2] for dimension-specific ranges
    ),
    dim: int,
    resolution: (
        tuple[float, ...]  # resolution per dimension
        | float  # same resolution for all dimensions
        | torch.Tensor  # tensor of densities
    ),
    batch_size: int = 1,
    margin: (
        tuple[tuple[float, float], ...]  # dims * [left_margin, right_margin]
        | tuple[float, ...]  # dims * margin (same left/right)
        | tuple[float, float]  # [left_margin, right_margin] for all dims
        | float  # same margin for all dims and sides
        | torch.Tensor  # shape [d, 2] for dimension-specific ranges
    ) = 0.0,
    multiple_of: (
        tuple[int, ...]  # the discretization length should be divisible per dimension
        | int  # same divisibily factor for all dimensions
        | torch.Tensor  # tensor of factors (containing integers)
    ) = 1,
    span_adjust_mode: (
        str | tuple[str, ...]  # 'both', 'left', or 'right'  # mode per dimension
    ) = "both",
    size_divisibility_adjust_mode: (
        str | tuple[str, ...]  # 'increase' or 'decrease'  # mode per dimension
    ) = "increase",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Construct a regular grid tensor of shape [batch_size, *grid_sizes, dim].

    Args:
        span: one of:
          - Tensor [dim,2] or [1,2] or [2] of (min, max) pairs
          - tuple of length 2 (min, max) to apply to all dims
          - tuple of length dim of (min, max) pairs
        dim: number of spatial dimensions d
        resolution: float or tuple of length d giving spacing between consecutive points
        batch_size: repeat the grid across leading batch dimension
        margin: non-negative margin(s) to add: same dispatch as span but per side
        multiple_of: int or tuple of length d: ensure each dim's point count is multiple-of
        span_adjust_mode: 'both'|'left'|'right' or tuple of length d for how to expand ranges
        size_divisibility_adjust_mode: 'increase'|'decrease' or tuple of length d for how to adjust
            n when not divisible by multiple_of
        device: torch.device (defaults to CPU)

    Returns:
        Tensor of shape [batch_size, *([n_i] for each dim i), dim]
    """

    # Hydra's ``_convert_="all"`` hands in plain Python ``list`` for YAML
    # sequences, but the normalization logic below switches on
    # ``isinstance(..., tuple)``. Promote lists to tuples recursively.
    def _lists_to_tuples(obj: Any) -> Any:
        if isinstance(obj, list):
            return tuple(_lists_to_tuples(item) for item in obj)
        return obj

    span = _lists_to_tuples(span)
    resolution = _lists_to_tuples(resolution)
    margin = _lists_to_tuples(margin)
    multiple_of = _lists_to_tuples(multiple_of)
    span_adjust_mode = _lists_to_tuples(span_adjust_mode)
    size_divisibility_adjust_mode = _lists_to_tuples(size_divisibility_adjust_mode)

    # --- Helper to normalize scalar/Tensor/tuple to tuple of length dim ---
    def _norm_param(param, name, expected_type, to_int=False):
        # Scalar
        if isinstance(param, expected_type):
            val = int(param) if to_int else float(param)
            return (val,) * dim
        # Tensor
        if isinstance(param, torch.Tensor):
            arr = param.squeeze().cpu().tolist()
            if len(arr) == 1:
                return (int(arr[0]) if to_int else float(arr[0]),) * dim
            elif len(arr) == dim:
                return tuple(int(v) if to_int else float(v) for v in arr)
        # tuple
        if isinstance(param, tuple) and len(param) == dim:
            out = []
            for v in param:
                if not isinstance(v, expected_type):
                    raise ValueError(
                        f"{name} values must be {expected_type.__name__},"
                        f" got {type(v)}"
                    )
                out.append(int(v) if to_int else float(v))
            return tuple(out)
        raise ValueError(
            f"{name} must be {expected_type.__name__} or " f"tuple of length {dim}"
        )

    resolutions = _norm_param(resolution, "resolution", (int, float))
    multiples = _norm_param(multiple_of, "multiple_of", int, to_int=True)

    # Validate multiple_of values are positive integers
    if not all(m > 0 for m in multiples):
        raise ValueError("multiple_of values must be positive integers")

    # --- Normalize span_adjust_mode ---
    span_modes = None
    valid_span_modes = {"both", "left", "right"}
    if isinstance(span_adjust_mode, str):
        if span_adjust_mode.lower() not in valid_span_modes:
            raise ValueError(f"span_adjust_mode must be one of {valid_span_modes}")
        span_modes = (span_adjust_mode,) * dim
    elif isinstance(span_adjust_mode, tuple) and len(span_adjust_mode) == dim:
        for m in span_adjust_mode:
            if m.lower() not in valid_span_modes:
                raise ValueError(
                    f"span_adjust_mode entries must be one of {valid_span_modes}"
                )
        span_modes = tuple(span_adjust_mode)
    else:
        raise ValueError(f"span_adjust_mode must be str or tuple of length {dim}")

    # --- Normalize size_divisibility_adjust_mode ---
    div_modes = None
    valid_div_modes = {"increase", "decrease"}
    if isinstance(size_divisibility_adjust_mode, str):
        if size_divisibility_adjust_mode.lower() not in valid_div_modes:
            raise ValueError(
                f"size_divisibility_adjust_mode must be one of {valid_div_modes}"
            )
        div_modes = (size_divisibility_adjust_mode,) * dim
    elif (
        isinstance(size_divisibility_adjust_mode, tuple)
        and len(size_divisibility_adjust_mode) == dim
    ):
        for m in size_divisibility_adjust_mode:
            if m.lower() not in valid_div_modes:
                raise ValueError(
                    f"size_divisibility_adjust_mode entries must be one of {valid_div_modes}"
                )
        div_modes = tuple(size_divisibility_adjust_mode)
    else:
        raise ValueError(
            f"size_divisibility_adjust_mode must be str or tuple of length {dim}"
        )

    # --- Normalize span to Tensor [dim,2] ---
    # Case: torch.Tensor
    if isinstance(span, torch.Tensor):
        rng = span.clone().to(device)
        if rng.ndim == 1 and rng.shape[0] == 2:
            rng = einops.repeat(rng, "lr -> d lr", d=dim, lr=2)
        elif rng.ndim == 2 and rng.shape == (1, 2):
            rng = einops.repeat(rng, "1 lr -> (1 d) lr", d=dim, lr=2)
        elif rng.ndim == 2 and rng.shape == (dim, 2):
            pass
        else:
            raise ValueError(
                f"span tensor must be shape [2], [1, 2], or [{dim}, 2], "
                f"got {tuple(rng.shape)}"
            )
    # Case: flat pair
    elif (
        isinstance(span, tuple)
        and len(span) == 2
        and all(isinstance(v, (int, float)) for v in span)
    ):
        rng = torch.tensor([span] * dim, dtype=torch.float32, device=device)
    # Case: per-dim list of pairs
    elif (
        isinstance(span, tuple)
        and len(span) == dim
        and all(
            isinstance(r, tuple)
            and len(r) == 2
            and all(isinstance(v, (int, float)) for v in r)
            for r in span
        )
    ):
        rng = torch.tensor(span, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported span type or shape for dimension {dim}")

    # Early sanity check
    if not torch.all(rng[:, 0] < rng[:, 1]):
        raise ValueError("All span min must be < max before margin")

    # re-use same logic as span but allowing scalar or pairs
    if isinstance(margin, (int, float)):
        margins = torch.full((dim, 2), float(margin), device=device)
    elif isinstance(margin, torch.Tensor):
        m = margin.clone().to(device)
        if m.ndim == 0:
            margins = einops.repeat(m, " -> d lr", d=dim, lr=2)
        elif m.ndim == 1 and m.shape[0] == 2:
            margins = einops.repeat(m, "lr -> d lr", d=dim, lr=2)
        elif m.ndim == 1 and m.shape[0] == dim:
            margins = einops.repeat(m, "d -> d lr", d=dim, lr=2)
        elif m.ndim == 2 and m.shape == (dim, 2):
            margins = m
        else:
            raise ValueError(f"margin tensor must be shape [2], [1, 2], or [{dim}, 2]")
    elif isinstance(margin, tuple):
        # length dim of scalars or pairs, or length-2 flat
        if len(margin) == 2 and all(isinstance(v, (int, float)) for v in margin):
            margins = torch.tensor([margin] * dim, device=device, dtype=torch.float32)
        elif len(margin) == dim and all(isinstance(v, (int, float)) for v in margin):
            margins = torch.tensor(
                [[v, v] for v in margin], device=device, dtype=torch.float32
            )
        elif len(margin) == dim and all(
            isinstance(v, tuple)
            and len(v) == 2
            and all(isinstance(x, (int, float)) for x in v)
            for v in margin
        ):
            margins = torch.tensor(margin, device=device, dtype=torch.float32)
        else:
            raise ValueError("Unsupported margin tuple format")
    else:
        raise ValueError(f"Unsupported margin type: {type(margin)}")

    if (margins < 0).any():
        raise ValueError("margin values must be non-negative")

    # apply margins
    rng[:, 0] -= margins[:, 0]
    rng[:, 1] += margins[:, 1]
    if not torch.all(rng[:, 0] < rng[:, 1]):
        raise ValueError("All span min must be < max after margin")

    # --- Build grid points per dimension ---
    grid_axes = []
    for i in range(dim):
        axis_min, axis_max = rng[i]
        res = float(resolutions[i])
        mf = multiples[i]
        spn_mode = span_modes[i].lower()
        div_mode = div_modes[i].lower()
        axis_span = axis_max - axis_min
        n = axis_span / res + 1

        # enforce multiple_of and being integer
        if n % mf != 0:
            if div_mode == "increase":
                adjusted_n = n + (mf - n % mf)
            else:  # div_mode == 'decrease'
                adjusted_n = n - (n % mf)
                # Ensure we have at least mf points
                if adjusted_n < mf:
                    adjusted_n = mf

            # Validate adjusted_n is positive
            if adjusted_n <= 0:
                raise ValueError(
                    f"Adjusted point count {adjusted_n} is not "
                    f"positive for dimension {i}."
                )
            # print(f'adjusted n from {n} to {adjusted_n} for multiple_of={mf}')

            # To maintain step size = res, while having adjusted_n points:
            # adjusted_n = extended_axis_span / res + 1
            # -> extended_axis_span = (adjusted_n - 1) * res
            extended_axis_span = (adjusted_n - 1) * res
            delta = extended_axis_span - axis_span
            if spn_mode == "both":
                axis_min -= delta / 2
                axis_max += delta / 2
            elif spn_mode == "left":
                axis_min -= delta
            else:
                axis_max += delta
            n = adjusted_n
            axis_span = axis_max - axis_min

        grid_axes.append(
            torch.linspace(axis_min, axis_max, steps=int(n), device=device)
        )

    # mesh and stack
    meshes = torch.meshgrid(*grid_axes, indexing="ij")
    grid = torch.stack(meshes, dim=-1)
    # add batch
    grid = einops.repeat(grid, "... d -> b ... d", b=batch_size, d=dim)
    return grid
