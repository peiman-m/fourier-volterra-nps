"""Batch-local step reducers — pure, stateless callables over a raw metric tensor.

Each reducer is invoked once per step with the raw per-metric tensor and
must return a scalar tensor — Lightning's ``self.log`` rejects anything
else. Epoch-level aggregation is the accumulator's job (see
``accumulators``), so reducers never decide across-batch semantics.

Shipping in v1: Mean / RMSEStep / Std / Median / Quantile / No.
Max/Min/Sum land when a concrete consumer needs them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseReducer(ABC):
    """Callable reducer collapsing a raw metric tensor to a scalar."""

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...


class MeanReducer(BaseReducer):
    def __call__(self, x: Tensor) -> Tensor:
        return x.mean()


class RMSEStepReducer(BaseReducer):
    """sqrt of mean — pair with ``squared_error`` metric_fn to get per-step RMSE."""

    def __call__(self, x: Tensor) -> Tensor:
        return torch.sqrt(x.mean())


class StdReducer(BaseReducer):
    """Biased (population) standard deviation — matches sample-accumulator convention."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.std(unbiased=False)


class MedianReducer(BaseReducer):
    def __call__(self, x: Tensor) -> Tensor:
        return x.median()


class QuantileReducer(BaseReducer):
    """Single-quantile reduction — e.g., ``QuantileReducer(q=0.95)``."""

    def __init__(self, q: float) -> None:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"QuantileReducer q must lie in [0, 1]; got {q}.")
        self.q = q

    def __call__(self, x: Tensor) -> Tensor:
        return torch.quantile(x, self.q)


class NoReducer(BaseReducer):
    """Pass-through — use when the metric function already returns a scalar."""

    def __call__(self, x: Tensor) -> Tensor:
        return x
