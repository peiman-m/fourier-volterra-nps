"""Loss callables for ``PhaseConfig.loss_fn`` — ``(model, batch) -> scalar Tensor``.

Kept distinct from the metric path. The wrapper's ``training_step``
calls ``cfg.loss_fn(model, batch)`` once per batch and ``backward()``s
on the result.

Both helpers route through ``as_batch`` and the forward-wrapper registry
so they work uniformly across batch types and model families.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from utils.data.base import BaseBatch, as_batch

from ..forward_wrappers import get_forward_wrapper


def nll_loss(model: nn.Module, batch: BaseBatch) -> Tensor:
    """Mean negative log-likelihood — the canonical NP training objective."""
    b = as_batch(batch, eval_on="query")
    fw = get_forward_wrapper(model, b)
    if fw is None:
        raise ValueError(
            f"No forward wrapper registered for "
            f"({type(model).__name__}, {type(b).__name__})."
        )
    likelihood = fw(model, b)
    return -likelihood.log_prob(b.yq).mean()


def mse_loss(model: nn.Module, batch: BaseBatch) -> Tensor:
    """Mean squared error of the predictive mean against ``batch.yq``."""
    b = as_batch(batch, eval_on="query")
    fw = get_forward_wrapper(model, b)
    if fw is None:
        raise ValueError(
            f"No forward wrapper registered for "
            f"({type(model).__name__}, {type(b).__name__})."
        )
    likelihood = fw(model, b)
    return ((likelihood.mean - b.yq) ** 2).mean()
