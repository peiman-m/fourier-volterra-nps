"""Epoch-level accumulators — ``torchmetrics.Metric`` subclasses.

DDP sync, device management, and state-dict participation come for free
from ``torchmetrics.Metric``. Three weighting families — the class-name
prefix names the aggregation mechanism:

- ``Sample*`` (``SampleMeanAccumulator`` / ``SampleStdAccumulator``) — each
  flattened per-point value contributes equally (pool every point, then
  reduce). Matches dataset-level mean/std when batch/task sizes vary.
- ``Batch*`` (``BatchMeanAccumulator`` / ``BatchStdAccumulator``) — each
  batch contributes equally regardless of size ("avg of batch means").
- ``Task*`` (``TaskMeanAccumulator``) — each *task* (one slice along the
  leading/batch dim) contributes equally, after averaging that task's own
  points. The meta-learning convention of "expected metric on a new task";
  robust to variable points-per-task.

Plus:

- ``CatAccumulator(finalize)`` — stores all raw values and applies
  ``finalize`` at ``compute()``. O(N) memory; needed for medians,
  quantiles, and anything non-associative whose streaming variant
  isn't worth maintaining.
- RMSE bundles a ``sqrt`` transform; the prefix names *where* the ``sqrt``
  sits: ``SampleRMSEAccumulator`` = ``sqrt`` of the global (pooled) mean
  squared-error; ``TaskRMSEAccumulator`` = mean over tasks of each task's
  own ``sqrt(mean SE)``. These differ even at equal task sizes (Jensen).

The three families diverge once per-batch/per-task point counts vary —
pick the one that matches what the result table is claiming.
"""
from __future__ import annotations

from typing import Callable

import torch
import torchmetrics
from torch import Tensor


class BaseAccumulator(torchmetrics.Metric):
    """Marker base class. Subclasses own state + update/compute."""

    # States are registered at runtime via ``add_state``; declare them here so
    # attribute access is typed concretely instead of nn.Module.__getattr__'s
    # ``Tensor | Module``. Not every subclass registers every state.
    sum: Tensor
    sum_sq: Tensor
    count: Tensor
    values: list[Tensor]


class SampleMeanAccumulator(BaseAccumulator):
    """Running mean where each flattened per-point value contributes equally."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        self.sum = self.sum + x.sum()
        self.count = self.count + x.numel()

    def compute(self) -> Tensor:
        return self.sum / self.count.to(self.sum.dtype)


class SampleStdAccumulator(BaseAccumulator):
    """Running population std where each flattened per-point value contributes equally."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        self.sum = self.sum + x.sum()
        self.sum_sq = self.sum_sq + (x ** 2).sum()
        self.count = self.count + x.numel()

    def compute(self) -> Tensor:
        count = self.count.to(self.sum.dtype)
        mean = self.sum / count
        var = self.sum_sq / count - mean ** 2
        return var.clamp_min(0.0).sqrt()


class BatchMeanAccumulator(BaseAccumulator):
    """Running mean where each *batch* contributes equally regardless of size."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        self.sum = self.sum + x.mean()
        self.count = self.count + 1

    def compute(self) -> Tensor:
        return self.sum / self.count.to(self.sum.dtype)


class BatchStdAccumulator(BaseAccumulator):
    """Running population std over per-batch scalar means (each batch one observation)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        m = x.mean()
        self.sum = self.sum + m
        self.sum_sq = self.sum_sq + m ** 2
        self.count = self.count + 1

    def compute(self) -> Tensor:
        count = self.count.to(self.sum.dtype)
        mean = self.sum / count
        var = self.sum_sq / count - mean ** 2
        return var.clamp_min(0.0).sqrt()


class TaskMeanAccumulator(BaseAccumulator):
    """Running mean where each *task* contributes equally, after averaging
    that task's own points.

    Reduces each batch ``[B, ...]`` to one scalar per task (mean over all
    non-leading point/output dims), then means those per-task values across
    every task in the epoch. Contrast: ``SampleMeanAccumulator`` weights every
    point equally; ``BatchMeanAccumulator`` weights every batch equally; this
    weights every *task* equally regardless of its point count — the
    meta-learning "expected metric on a new task". Use for additive per-point
    metrics such as ``log_likelihood`` and ``gaussian_crps_closed_form``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @staticmethod
    def _per_task(x: Tensor) -> Tensor:
        """``[B, ...] -> [B]`` by mean over non-leading dims; pass 1-D through."""
        return x.flatten(start_dim=1).mean(dim=1) if x.ndim > 1 else x.reshape(-1)

    def update(self, x: Tensor) -> None:
        per_task = self._per_task(x)
        self.sum = self.sum + per_task.sum()
        self.count = self.count + per_task.numel()

    def compute(self) -> Tensor:
        return self.sum / self.count.to(self.sum.dtype)


class TaskStdAccumulator(BaseAccumulator):
    """Population std over per-task values (each task one observation).

    Reduces each batch ``[B, ...]`` to one scalar per task (mean over non-leading
    point/output dims), then takes the population std of those per-task values
    across all tasks. Pairs with ``TaskMeanAccumulator`` to report
    "mean ± std over tasks" — the task-level spread / error bar. Contrast
    ``SampleStdAccumulator`` (spread over individual points) and
    ``BatchStdAccumulator`` (spread over per-batch means). Intended for additive
    per-point metrics such as ``log_likelihood`` and ``gaussian_crps_closed_form``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        per_task = TaskMeanAccumulator._per_task(x)
        self.sum = self.sum + per_task.sum()
        self.sum_sq = self.sum_sq + (per_task ** 2).sum()
        self.count = self.count + per_task.numel()

    def compute(self) -> Tensor:
        count = self.count.to(self.sum.dtype)
        mean = self.sum / count
        var = self.sum_sq / count - mean ** 2
        return var.clamp_min(0.0).sqrt()


class CatAccumulator(BaseAccumulator):
    """Stores all raw values (flattened); applies ``finalize`` at ``compute``.

    O(N) memory; the right choice for medians, quantiles, and anything
    non-associative that needs the full distribution.
    """

    # Tell torchmetrics this is a list-valued state so it handles cat-concat across DDP ranks.
    full_state_update = False

    def __init__(self, finalize: Callable[[Tensor], Tensor], **kwargs) -> None:
        super().__init__(**kwargs)
        self.finalize = finalize
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, x: Tensor) -> None:
        self.values.append(x.detach().flatten())

    def compute(self) -> Tensor:
        if not self.values:
            return torch.tensor(float("nan"))
        # torchmetrics may already concatenate list states across DDP ranks into a single Tensor.
        flat = torch.cat(self.values) if isinstance(self.values, list) else self.values
        return self.finalize(flat)


class SampleRMSEAccumulator(SampleMeanAccumulator):
    """Global RMSE: ``sqrt`` of the *sample-mean* of squared errors.

    Pools squared errors over every point in the epoch, takes their mean
    (the global MSE), then ``sqrt``. Expects squared-error inputs (e.g.,
    ``utils.experiment.metrics.functions.squared_error``).
    """

    def compute(self) -> Tensor:
        return super().compute().clamp_min(0.0).sqrt()


class TaskRMSEAccumulator(BaseAccumulator):
    """Task-weighted RMSE: mean over tasks of each task's own ``sqrt(mean SE)``.

    Reduces each batch ``[B, ...]`` of squared errors to one RMSE per task
    (``sqrt`` of the mean SE over that task's points), then averages those
    per-task RMSEs across all tasks (each task equal weight). Because of the
    per-task ``sqrt`` this is *not* equal to ``SampleRMSEAccumulator`` even
    when every task has the same number of points (Jensen's inequality).
    Expects squared-error inputs (``...functions.squared_error``).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        per_task_mse = (
            x.flatten(start_dim=1).mean(dim=1) if x.ndim > 1 else x.reshape(-1)
        )
        per_task_rmse = per_task_mse.clamp_min(0.0).sqrt()
        self.sum = self.sum + per_task_rmse.sum()
        self.count = self.count + per_task_rmse.numel()

    def compute(self) -> Tensor:
        return self.sum / self.count.to(self.sum.dtype)
