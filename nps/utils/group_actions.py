import torch


def translation(
    x1: torch.Tensor,
    x2: torch.Tensor,
    diagonal: bool = False,
) -> torch.Tensor:
    if not diagonal:
        return x1[..., :, None, :] - x2[..., None, :, :]

    assert x1.shape == x2.shape, "Must be the same shape."
    return x1 - x2