import torch


def sq_dist(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute the weights for the SetConv layer,
        mapping from `x1` to `x2`.

    Arguments:
        x1: Tensor of shape (batch_size, num_x1, dim)
        x2: Tensor of shape (batch_size, num_x2, dim)
        lengthscales: Tensor of shape (dim,) or (dim, num_lengthscales)

    Returns:
        Tensor of shape (batch_size, num_x1, num_x2, dim)
    """

    x1_ = x1[..., None, :]
    x2_ = x2[..., None, :, :]
    return (x1_ - x2_).pow(2)