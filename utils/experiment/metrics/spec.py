"""MetricSpec — the composed (metric_fn, accumulator, step_reducer, eval_on) dataclass.

Three fields are what a user actually sets per metric (``metric_fn``,
``name``, ``accumulator``); everything else is defaults. Epoch-side
reduction is the accumulator's job; there is no post-compute reduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import Tensor
from torch.distributions import Distribution

from utils.data.base import BaseBatch, EvalOn

from .accumulators import BaseAccumulator
from .reducers import BaseReducer


MetricFn = Callable[[Distribution, BaseBatch], Tensor]


@dataclass
class MetricSpec:
    """Spec for one metric tracked through the Lightning loop.

    ``step_reducer=None`` disables step-level logging for this metric —
    only the accumulator's epoch value is logged. ``prog_bar`` and
    ``sync_dist`` apply to the step-level log call only; they are
    nonsensical without a ``step_reducer`` and ``__post_init__``
    rejects that combination.

    ``eval_on`` tags whether the metric scores on query points or on
    context points. The Lightning loop groups specs by this value and
    calls the forward wrapper once per group — at most two forward
    calls per step in current scope.
    """

    metric_fn: MetricFn
    name: str
    accumulator: BaseAccumulator
    step_reducer: BaseReducer | None = None
    eval_on: EvalOn = "query"
    prog_bar: bool = False
    sync_dist: bool = False

    def __post_init__(self) -> None:
        if self.step_reducer is None:
            if self.prog_bar:
                raise ValueError(
                    f"MetricSpec(name={self.name!r}): prog_bar=True requires a step_reducer; "
                    "without one, there is no step-level value to show on the progress bar."
                )
            if self.sync_dist:
                raise ValueError(
                    f"MetricSpec(name={self.name!r}): sync_dist=True requires a step_reducer; "
                    "epoch-level sync is handled by the accumulator itself."
                )
