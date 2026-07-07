"""Forward wrappers — bridge ``model.forward`` to a uniform ``(model, batch) -> Distribution`` shape.

The wrapper is dispatched on ``(type(model), type(batch))`` via the
registry. Inside ``_run_step`` the caller has already run
``as_batch(batch, eval_on=...)`` so ``batch.xq`` / ``batch.yq`` /
``batch.mq_grid`` already point at the correct held-out slot — wrappers
no longer parse a ``query_subset`` kwarg.

``Batch`` and flat batches share the same attribute names
(``xc``/``yc``/``xq``/``yq``), so one wrapper is registered against both
batch families instead of separate ``cnp_forward_wrapper`` /
``flat_xy_cnp_forward_wrapper`` entries. ``GridConvCNP`` keeps its own
wrapper because its ``forward`` takes ``y_mc`` / ``y`` / ``y_mq`` grid
tensors.
"""
from collections.abc import Callable
from typing import Type, cast

import torch.nn as nn
from torch.distributions import Distribution

from nps.models import *

from ..data import *
from .registry.base import (
    BaseWrapperRegistry,
    register_class_wrapper,
    register_instance_wrapper,
)


ModelForwardWrapper = Callable[[nn.Module, BaseBatch], Distribution]

registry = BaseWrapperRegistry[ModelForwardWrapper]()


def register_forward_wrapper(
    model_cls: Type[nn.Module] | tuple[Type[nn.Module], ...],
    batch_cls: Type[BaseBatch] | tuple[Type[BaseBatch], ...],
) -> Callable[[ModelForwardWrapper], ModelForwardWrapper]:
    return register_class_wrapper(registry, model_cls, batch_cls)


def register_instance_forward_wrapper(
    model: nn.Module,
) -> Callable[[ModelForwardWrapper], ModelForwardWrapper]:
    return register_instance_wrapper(registry, model)


def get_forward_wrapper(
    model: nn.Module, batch: BaseBatch
) -> ModelForwardWrapper | None:
    return registry.get_wrapper(model, batch)


@register_forward_wrapper(
    (CNP, ACNP, TNP, TETNP, ConvCNP, SetFourierConvCNP),
    (Batch, SyntheticBatch, PredPreyBatch, ImageBatch, KolmogorovBatch, ERA5Batch),
)
def cnp_forward_wrapper(model: nn.Module, batch: BaseBatch) -> Distribution:
    return model(xc=batch.xc, yc=batch.yc, xq=batch.xq)


@register_forward_wrapper(
    GridConvCNP,
    (ImageBatch, KolmogorovBatch, ERA5Batch),
)
def grid_xy_cnp_forward_wrapper(model: nn.Module, batch: BaseBatch) -> Distribution:
    # Registered only for grid batches, which carry the grid fields.
    grid = cast(ImageBatch | KolmogorovBatch | ERA5Batch, batch)
    return model(y_mc=grid.y_mc_grid, y=grid.y_grid, y_mq=grid.mq_grid)
