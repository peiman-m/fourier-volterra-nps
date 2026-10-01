import copy
import math
import warnings
from collections.abc import Callable

import torch
import torch.nn as nn

_warned_once: set[str] = set()


def warn_once(message: str, category: type[Warning] = UserWarning) -> None:
    """Emit ``message`` via ``warnings.warn`` at most once per process.

    Guards against log flooding when a warning would otherwise fire on
    every forward pass (e.g. an unused ``mask`` passed every training
    step). Deduplicates on the message text, independent of the global
    warning-filter state.
    """
    if message in _warned_once:
        return
    _warned_once.add(message)
    warnings.warn(message, category, stacklevel=2)


def preprocess_observations(
    xq: torch.Tensor,
    yc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Processes the observations by appending an additional density channel:
    - Unobserved query outputs (yq) are initialized as zero tensors and
        assigned a density channel of 1.
    - Observed values (yc) are assigned a density channel of 0.

    Args:
        xq (torch.Tensor): Query inputs.
        yc (torch.Tensor): Context outputs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            Processed yc and yq with appended density channels.
    """
    device, dtype = yc.device, yc.dtype
    kwargs = {"device": device, "dtype": dtype}

    # Create yq with same spatial shape as xq, but last dim matching yc
    yq_shape = xq.shape[:-1] + (yc.shape[-1],)
    yq = torch.zeros(yq_shape, **kwargs)

    # Density channel shape
    yc_density_shape = yc.shape[:-1] + (1,)
    yq_density_shape = yq.shape[:-1] + (1,)

    # Append density channel (0 for yc, 1 for yq)
    yc = torch.cat((yc, torch.zeros(yc_density_shape, **kwargs)), dim=-1)
    yq = torch.cat((yq, torch.ones(yq_density_shape, **kwargs)), dim=-1)

    return yc, yq


def get_clones(
    module: nn.Module,
    n: int,
) -> nn.ModuleList:
    """Create N identical layers.
    Args:
        module (nn.Module): Module to clone.
        n (int): Number of clones.
    Returns:
        nn.ModuleList: List of cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def compress_batch_dimensions(
    x: torch.Tensor, other_dims: int
) -> tuple[torch.Tensor, Callable]:
    """Compress multiple batch dimensions of a tensor into a single batch dimension.

    Args:
        x (tensor): Tensor to compress.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tensor: `x` with batch dimensions compressed.
        function: Function to undo the compression of the batch dimensions.
    """
    batch_shape = x.shape[:-other_dims]

    if len(batch_shape) == 1:
        return x, lambda x: x
    else:

        def uncompress(x_after):
            return x_after.reshape(*batch_shape, *x_after.shape[1:])

        compressed_shape = (math.prod(batch_shape),) + x.shape[-other_dims:]
        return x.reshape(compressed_shape), uncompress


def convert(value, type_):
    """Convert a value to a specified type if it isn't already of that type.

    Args:
        value: The value to potentially convert.
        type_: The type to convert to.

    Returns:
        The value converted to the specified type if it wasn't already,
        or the original value if it was already of the specified type.
    """
    if isinstance(value, type_):
        return value
    else:
        return type_((value,))
